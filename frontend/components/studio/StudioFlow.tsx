"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/hooks/useLanguage";
import { useRecorder } from "@/hooks/useRecorder";
import { processVocal } from "@/lib/api";
import { AutotuneSelector, type AutotuneLevel } from "./AutotuneSelector";
import { ProcessingStatus } from "./ProcessingStatus";
import { LevelMeter } from "@/components/voice-training/LevelMeter";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

type Step = "setup" | "recording" | "review" | "processing" | "done" | "error";

export function StudioFlow() {
  const router = useRouter();
  const { language, isRTL } = useLanguage();
  const { state, audioBlob, audioUrl, levels, duration, clipping, tooQuiet, error, startRecording, stopRecording, reset } = useRecorder();

  const [step, setStep] = useState<Step>("setup");
  const [projectName, setProjectName] = useState("");
  const [autotuneLevel, setAutotuneLevel] = useState<AutotuneLevel>("subtle");
  const [processedUrl, setProcessedUrl] = useState("");
  const [projectId, setProjectId] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [resultMsg, setResultMsg] = useState("");

  const t = {
    songName:       language === "ar" ? "اسم الأغنية" : "Song Name",
    namePlaceholder:language === "ar" ? "أغنيتي الأولى" : "My First Song",
    recordYourVocal:language === "ar" ? "سجّل صوتك" : "Record Your Vocal",
    startRecording: language === "ar" ? "ابدأ التسجيل" : "Start Recording",
    stop:           language === "ar" ? "إيقاف" : "Stop",
    reRecord:       language === "ar" ? "إعادة التسجيل" : "Re-record",
    processVocal:   language === "ar" ? "معالجة الصوت" : "Process My Vocal",
    recording:      language === "ar" ? "جار التسجيل" : "Recording",
    listenBack:     language === "ar" ? "استمع للتسجيل" : "Listen back",
    processed:      language === "ar" ? "صوتك المعالج" : "Your Processed Vocal",
    nextStep:       language === "ar" ? "توليد الإيقاع" : "Generate My Beat",
    tryAgain:       language === "ar" ? "المحاولة مجدداً" : "Try Again",
  };

  async function handleProcess() {
    if (!audioBlob) return;
    setStep("processing");
    try {
      const result = await processVocal(
        audioBlob,
        projectName.trim() || (language === "ar" ? "أغنيتي" : "My Song"),
        autotuneLevel,
        language,
      );
      setProcessedUrl(result.processed_url);
      setProjectId(result.project_id);
      setResultMsg(language === "ar" ? result.message_ar : result.message_en);
      setStep("done");
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string; debug?: string } | string };
      const detail = e?.detail;
      if (detail && typeof detail === "object" && detail.message_en) {
        setErrorMsg(language === "ar" ? (detail.message_ar ?? detail.message_en) : detail.message_en);
      } else {
        setErrorMsg(language === "ar" ? "حدث خطأ. يرجى المحاولة مرة أخرى." : "Something went wrong. Please try again.");
      }
      setStep("error");
    }
  }

  const formatDuration = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  // ── setup ────────────────────────────────────────────────────────────────
  if (step === "setup") {
    return (
      <div className="flex flex-col gap-6 w-full max-w-lg" dir={isRTL ? "rtl" : "ltr"}>
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-white">
            {language === "ar" ? "أغنية جديدة" : "New Song"}
          </h1>
          <p className="text-zinc-400 text-sm">
            {language === "ar"
              ? "سجّل صوتك وكلماتك — الذكاء الاصطناعي يتولى الباقي."
              : "Record your voice and your lyrics — the AI handles the rest."}
          </p>
        </div>

        <Input
          label={t.songName}
          placeholder={t.namePlaceholder}
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          maxLength={60}
        />

        <AutotuneSelector value={autotuneLevel} onChange={setAutotuneLevel} />

        <Button fullWidth onClick={() => setStep("recording")}>
          <MicIcon />
          {t.recordYourVocal}
        </Button>
      </div>
    );
  }

  // ── recording ────────────────────────────────────────────────────────────
  if (step === "recording") {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-lg" dir={isRTL ? "rtl" : "ltr"}>
        <div className="text-center">
          <h2 className="text-xl font-bold text-white">
            {language === "ar" ? "سجّل أغنيتك" : "Sing Your Song"}
          </h2>
          <p className="text-sm text-zinc-400 mt-1">
            {language === "ar"
              ? "غنّ بصوتك الطبيعي — سنتولى المعالجة"
              : "Sing naturally — we handle the processing"}
          </p>
        </div>

        <LevelMeter levels={levels} isRecording={state === "recording"} />

        {state === "recording" && (
          <div className="flex items-center gap-2 text-sm text-red-400">
            <span className="h-2 w-2 rounded-full bg-red-400 animate-pulse" />
            {t.recording} • {formatDuration(duration)}
          </div>
        )}

        {state === "recording" && clipping && (
          <p className="text-sm text-red-400 text-center font-medium">
            {language === "ar"
              ? "الصوت مرتفع جداً ويتشوه — ابتعد قليلاً عن الميكروفون"
              : "Too loud — your recording is distorting. Move back from the mic."}
          </p>
        )}
        {state === "recording" && !clipping && tooQuiet && (
          <p className="text-sm text-amber-400 text-center">
            {language === "ar"
              ? "صوتك منخفض — اقترب من الميكروفون"
              : "A bit quiet — move closer to the mic."}
          </p>
        )}

        {(state === "error" || error) && (
          <p className="text-sm text-red-400 text-center">{error}</p>
        )}

        <div className="w-full space-y-3">
          {(state === "idle" || state === "error") && (
            <Button fullWidth onClick={startRecording}>
              <MicIcon /> {t.startRecording}
            </Button>
          )}
          {state === "requesting" && <Button fullWidth loading>{t.startRecording}</Button>}
          {state === "recording" && (
            <Button fullWidth variant="secondary" onClick={stopRecording}>
              <StopIcon /> {t.stop}
            </Button>
          )}
          {state === "stopped" && audioUrl && (
            <div className="space-y-3">
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3">
                <p className="text-xs text-zinc-500 mb-2">{t.listenBack}</p>
                <audio controls src={audioUrl} className="w-full h-10" />
              </div>
              <div className="flex gap-3">
                <Button variant="secondary" className="flex-1" onClick={() => { reset(); }}>
                  {t.reRecord}
                </Button>
                <Button className="flex-1" onClick={handleProcess}>
                  {t.processVocal}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── processing ───────────────────────────────────────────────────────────
  if (step === "processing") {
    return <ProcessingStatus />;
  }

  // ── done ─────────────────────────────────────────────────────────────────
  if (step === "done") {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-lg text-center" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-16 w-16 rounded-full bg-violet-500/20 flex items-center justify-center text-3xl">✨</div>
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">{t.processed}</h2>
          <p className="text-sm text-zinc-400">{resultMsg}</p>
        </div>

        <div className="w-full rounded-xl border border-zinc-800 bg-zinc-900 p-4 space-y-2">
          <p className="text-xs text-zinc-500 text-left">
            {language === "ar" ? "الإصدار المعالج" : "Processed version"}
          </p>
          <audio controls src={processedUrl} className="w-full h-10" />
        </div>

        <Button fullWidth onClick={() => router.push(`/studio/${projectId}/beat`)}>
          {t.nextStep} →
        </Button>
        <button
          className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
          onClick={() => { reset(); setStep("recording"); }}
        >
          {language === "ar" ? "إعادة تسجيل الصوت" : "Re-record vocal instead"}
        </button>
      </div>
    );
  }

  // ── error ─────────────────────────────────────────────────────────────────
  if (step === "error") {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-sm text-center" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-16 w-16 rounded-full bg-red-500/20 flex items-center justify-center text-3xl">⚠️</div>
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">
            {language === "ar" ? "مشكلة في المعالجة" : "Processing issue"}
          </h2>
          <p className="text-sm text-zinc-400 whitespace-pre-wrap">{errorMsg}</p>
        </div>
        <div className="flex gap-3 w-full">
          <Button variant="secondary" className="flex-1" onClick={() => router.push("/dashboard")}>
            {language === "ar" ? "العودة" : "Go Back"}
          </Button>
          {/* The take is precious — a server hiccup must never force a re-sing.
              Retry re-sends the SAME recording; re-recording is the fallback. */}
          {audioBlob ? (
            <Button className="flex-1" onClick={handleProcess}>
              {t.tryAgain}
            </Button>
          ) : (
            <Button className="flex-1" onClick={() => { reset(); setStep("recording"); }}>
              {t.reRecord}
            </Button>
          )}
        </div>
        {audioBlob && (
          <button
            className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
            onClick={() => { reset(); setStep("recording"); }}
          >
            {language === "ar" ? "تسجيل أداء جديد بدلاً من ذلك" : "Record a new take instead"}
          </button>
        )}
      </div>
    );
  }

  return null;
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
