"use client";

import { useState, useRef, useCallback, useEffect } from "react";

export type RecorderState = "idle" | "requesting" | "recording" | "stopped" | "error";

export interface RecorderResult {
  state: RecorderState;
  audioBlob: Blob | null;
  audioUrl: string | null;
  levels: number[];       // 0–1 values for level meter (32 bars)
  duration: number;       // seconds recorded so far
  clipping: boolean;      // true while the input is hitting digital ceiling
  tooQuiet: boolean;      // true when the sung level has stayed very low
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  reset: () => void;
}

const BARS = 32;
// Time-domain thresholds for live input coaching. Peak near full-scale for a
// few frames = clipping; running RMS staying under the floor while recording
// = the singer is too far from the mic.
const CLIP_PEAK = 0.97;
const CLIP_HOLD_FRAMES = 3;
const QUIET_RMS = 0.01;
const QUIET_HOLD_MS = 4000;

export function useRecorder(): RecorderResult {
  const [state, setState] = useState<RecorderState>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [levels, setLevels] = useState<number[]>(Array(BARS).fill(0));
  const [duration, setDuration] = useState(0);
  const [clipping, setClipping] = useState(false);
  const [tooQuiet, setTooQuiet] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const stopAnimations = useCallback(() => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setState("requesting");
      setError(null);

      // Music-grade capture: the browser's echoCancellation / noiseSuppression /
      // autoGainControl are telephony processors tuned for speech calls — they
      // gate sustained sung notes, chew vibrato and breath tails, and pump the
      // dynamics the beat engine reacts to. The vocal IS the composition:
      // capture it honestly and let the backend chain do all cleanup.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: 48000,
        },
      });
      streamRef.current = stream;

      // Set up Web Audio analyser for level meter
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Pick best supported format
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      // This recording is the master source for the whole song — encode Opus
      // well above the default voice-call bitrate.
      const recorder = new MediaRecorder(stream, { mimeType, audioBitsPerSecond: 256000 });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioUrl(url);
        stopAnimations();
        setLevels(Array(BARS).fill(0));

        // Stop mic stream
        streamRef.current?.getTracks().forEach((t) => t.stop());
      };

      recorder.start(100); // collect chunks every 100ms
      setState("recording");
      startTimeRef.current = Date.now();

      // Duration timer
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 500);

      // Level meter animation + live input coaching (real peak/RMS from the
      // time-domain waveform — the frequency bars are decoration, not a meter)
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const waveArray = new Float32Array(analyser.fftSize);
      let clipFrames = 0;
      let loudSince = performance.now();
      const animate = () => {
        analyser.getByteFrequencyData(dataArray);
        // Downsample to BARS buckets
        const bucketSize = Math.floor(dataArray.length / BARS);
        const bars = Array.from({ length: BARS }, (_, i) => {
          const slice = dataArray.slice(i * bucketSize, (i + 1) * bucketSize);
          const avg = slice.reduce((s, v) => s + v, 0) / slice.length;
          return avg / 255;
        });
        setLevels(bars);

        analyser.getFloatTimeDomainData(waveArray);
        let peak = 0;
        let sumSq = 0;
        for (let i = 0; i < waveArray.length; i++) {
          const v = Math.abs(waveArray[i]);
          if (v > peak) peak = v;
          sumSq += waveArray[i] * waveArray[i];
        }
        const rms = Math.sqrt(sumSq / waveArray.length);
        clipFrames = peak >= CLIP_PEAK ? clipFrames + 1 : 0;
        setClipping(clipFrames >= CLIP_HOLD_FRAMES);
        const now = performance.now();
        if (rms >= QUIET_RMS) loudSince = now;
        setTooQuiet(now - loudSince > QUIET_HOLD_MS);

        animFrameRef.current = requestAnimationFrame(animate);
      };
      animate();
    } catch (err) {
      setState("error");
      setError(err instanceof Error && err.name === "NotAllowedError"
        ? "Microphone permission denied. Please allow mic access and try again."
        : "Could not access microphone. Please check your settings.");
    }
  }, [stopAnimations]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setState("stopped");
      setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, []);

  const reset = useCallback(() => {
    stopAnimations();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setState("idle");
    setAudioBlob(null);
    setAudioUrl(null);
    setLevels(Array(BARS).fill(0));
    setDuration(0);
    setClipping(false);
    setTooQuiet(false);
    setError(null);
    chunksRef.current = [];
  }, [audioUrl, stopAnimations]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAnimations();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl, stopAnimations]);

  return { state, audioBlob, audioUrl, levels, duration, clipping, tooQuiet, error, startRecording, stopRecording, reset };
}
