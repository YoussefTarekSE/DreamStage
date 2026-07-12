"use client";

import { useEffect } from "react";
import { useRecorder } from "@/hooks/useRecorder";
import { LevelMeter } from "./LevelMeter";
import { Button } from "@/components/ui/Button";
import { useLanguage } from "@/hooks/useLanguage";

interface PhraseRecorderProps {
  phrase: string;
  phraseIndex: number;
  totalPhrases: number;
  onAccept: (blob: Blob) => void;
  isRTL: boolean;
}

export function PhraseRecorder({
  phrase,
  phraseIndex,
  totalPhrases,
  onAccept,
  isRTL,
}: PhraseRecorderProps) {
  const { language } = useLanguage();
  const { state, audioBlob, audioUrl, levels, duration, error, startRecording, stopRecording, reset } =
    useRecorder();

  // Auto-stop after 30 seconds
  useEffect(() => {
    if (duration >= 30) stopRecording();
  }, [duration, stopRecording]);

  const formatDuration = (s: number) => `${s}s`;

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-lg" dir={isRTL ? "rtl" : "ltr"}>
      {/* Progress */}
      <div className="flex items-center gap-2">
        {Array.from({ length: totalPhrases }).map((_, i) => (
          <div
            key={i}
            className={`h-1.5 w-8 rounded-full transition-colors ${
              i < phraseIndex ? "bg-violet-500" : i === phraseIndex ? "bg-violet-400" : "bg-zinc-700"
            }`}
          />
        ))}
      </div>

      {/* Phrase counter */}
      <p className="text-xs text-zinc-500 uppercase tracking-wider">
        {language === "ar"
          ? `العبارة ${phraseIndex + 1} من ${totalPhrases}`
          : `Phrase ${phraseIndex + 1} of ${totalPhrases}`}
      </p>

      {/* The lyric phrase */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 px-8 py-6 text-center w-full">
        <p className="text-xl font-medium text-white leading-relaxed">{phrase}</p>
        <p className="mt-2 text-sm text-zinc-500">
          {language === "ar" ? "غنّ هذه العبارة بصوتك الطبيعي" : "Sing this phrase in your natural voice"}
        </p>
      </div>

      {/* Level meter */}
      <LevelMeter levels={levels} isRecording={state === "recording"} />

      {/* Duration */}
      {state === "recording" && (
        <div className="flex items-center gap-2 text-sm text-red-400">
          <span className="h-2 w-2 rounded-full bg-red-400 animate-pulse" />
          {language === "ar" ? `جار التسجيل • ${formatDuration(duration)}` : `Recording • ${formatDuration(duration)}`}
        </div>
      )}

      {/* Error */}
      {(state === "error" || error) && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400 text-center">
          {error}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col gap-3 w-full">
        {state === "idle" || state === "error" ? (
          <Button fullWidth onClick={startRecording}>
            <MicIcon />
            {language === "ar" ? "ابدأ التسجيل" : "Start Recording"}
          </Button>
        ) : state === "requesting" ? (
          <Button fullWidth loading>
            {language === "ar" ? "جار طلب الإذن..." : "Requesting mic permission..."}
          </Button>
        ) : state === "recording" ? (
          <Button fullWidth variant="secondary" onClick={stopRecording}>
            <StopIcon />
            {language === "ar" ? "إيقاف" : "Stop"}
          </Button>
        ) : state === "stopped" && audioUrl ? (
          <div className="flex flex-col gap-3 w-full">
            {/* Playback */}
            <audio controls src={audioUrl} className="w-full h-10 rounded-lg" />

            <div className="flex gap-3">
              <Button variant="secondary" onClick={reset} className="flex-1">
                {language === "ar" ? "إعادة التسجيل" : "Re-record"}
              </Button>
              <Button
                className="flex-1"
                onClick={() => audioBlob && onAccept(audioBlob)}
              >
                {language === "ar" ? "استخدم هذا" : "Use This"}
              </Button>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function MicIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}
