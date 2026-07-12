"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/hooks/useLanguage";
import { createMix, getDownloadUrls, submitProjectFeedback } from "@/lib/api";
import { Button } from "@/components/ui/Button";

type Step = "loading_cached" | "mixing" | "done" | "error";

interface MixFlowProps {
  projectId: string;
  projectName: string;
  cachedMp3Url?: string;
  cachedWavUrl?: string;
  hasCachedMix?: boolean;
}

const MIXING_STEPS_EN = [
  "Loading your vocal and beat...",
  "Carving space for your voice in the mix...",
  "Adding room reverb to the vocal...",
  "Widening the beat to stereo...",
  "Blending vocal and instrumental...",
  "Applying mastering chain...",
  "Limiting and loudness normalizing...",
  "Encoding MP3 320kbps + WAV 24-bit...",
  "Almost there...",
];
const MIXING_STEPS_AR = [
  "تحميل صوتك والإيقاع...",
  "تهيئة المساحة الصوتية...",
  "إضافة صدى خفيف للصوت...",
  "توسيع الإيقاع إلى ستيريو...",
  "مزج الصوت والموسيقى...",
  "تطبيق سلسلة الماسترنج...",
  "ضبط مستوى الصوت النهائي...",
  "تصدير MP3 + WAV...",
  "لحظة أخيرة...",
];

export function MixFlow({ projectId, projectName, cachedMp3Url, cachedWavUrl, hasCachedMix = false }: MixFlowProps) {
  const router = useRouter();
  const { language, isRTL } = useLanguage();

  const [step, setStep] = useState<Step>(cachedMp3Url ? "done" : hasCachedMix ? "loading_cached" : "mixing");
  const [mp3Url, setMp3Url] = useState(cachedMp3Url ?? "");
  const [wavUrl, setWavUrl] = useState(cachedWavUrl ?? "");
  const [errorMsg, setErrorMsg] = useState("");
  const [mixMsgIdx, setMixMsgIdx] = useState(0);
  const [feedback, setFeedback] = useState({ beat_quality: 0, vocal_preservation: 0, overall_satisfaction: 0 });
  const [feedbackSent, setFeedbackSent] = useState(false);
  const steps = language === "ar" ? MIXING_STEPS_AR : MIXING_STEPS_EN;

  // Cycle through mixing messages
  useEffect(() => {
    if (step !== "mixing") return;
    const t = setInterval(() => setMixMsgIdx(i => Math.min(i + 1, steps.length - 1)), 3500);
    return () => clearInterval(t);
  }, [step, steps.length]);

  // Auto-start mixing
  useEffect(() => {
    if (cachedMp3Url) return;
    if (hasCachedMix) {
      loadCachedMix();
    } else {
      runMix();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadCachedMix() {
    try {
      const result = await getDownloadUrls(projectId);
      setMp3Url(result.mp3_url ?? "");
      setWavUrl(result.wav_url ?? "");
      setStep("done");
    } catch {
      setErrorMsg("Could not load the completed mix.");
      setStep("error");
    }
  }

  async function runMix() {
    setStep("mixing");
    try {
      const result = await createMix(projectId);
      setMp3Url(result.mp3_url);
      setWavUrl(result.wav_url);
      setStep("done");
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
      const d = e?.detail;
      if (d && typeof d === "object" && d.message_en) {
        setErrorMsg(language === "ar" ? (d.message_ar ?? d.message_en) : d.message_en);
      } else {
        setErrorMsg(language === "ar" ? "فشل المزج. يرجى المحاولة مرة أخرى." : "Mixing failed. Please try again.");
      }
      setStep("error");
    }
  }

  async function sendFeedback() {
    if (!feedback.beat_quality || !feedback.vocal_preservation || !feedback.overall_satisfaction) return;
    try {
      await submitProjectFeedback(projectId, feedback);
      setFeedbackSent(true);
    } catch {
      setFeedbackSent(false);
    }
  }

  // ── Mixing ────────────────────────────────────────────────────────────────
  if (step === "loading_cached") {
    return (
      <div className="flex flex-col items-center gap-4 text-center py-8">
        <div className="h-10 w-10 rounded-full border-2 border-t-violet-500 border-zinc-800 animate-spin" />
        <p className="text-sm text-zinc-400">
          {language === "ar" ? "جارٍ تحميل الميكس..." : "Loading mix..."}
        </p>
      </div>
    );
  }

  if (step === "mixing") {
    return (
      <div className="flex flex-col items-center gap-8 py-8 text-center">
        <div className="relative h-28 w-28">
          <div className="absolute inset-0 rounded-full border-2 border-violet-500/10 animate-ping" />
          <div className="absolute inset-0 rounded-full border-2 border-violet-500/20" />
          <div className="absolute inset-2 rounded-full border-2 border-t-violet-500 animate-spin" />
          <div className="absolute inset-5 rounded-full border-2 border-t-violet-300/60 animate-spin" style={{ animationDirection: "reverse", animationDuration: "1.8s" }} />
          <div className="absolute inset-0 flex items-center justify-center text-3xl">🎚️</div>
        </div>

        <div className="space-y-3">
          <h2 className="text-2xl font-bold text-white">
            {language === "ar" ? "جار المزج الاحترافي" : "Professional Mixing"}
          </h2>
          <p className="text-sm text-violet-400 animate-pulse min-h-[1.25rem]">
            {steps[mixMsgIdx]}
          </p>
        </div>

        <div className="max-w-sm space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-left">
          {[
            language === "ar" ? "ستيريو بتأثير هاس" : "Haas stereo widening",
            language === "ar" ? "قطع EQ للصوت" : "EQ carve for vocal space",
            language === "ar" ? "ضغط متعدد النطاقات" : "Multiband compression",
            language === "ar" ? "Mid-Side تحسين" : "Mid-Side enhancement",
            language === "ar" ? "حد True-Peak" : "True-peak limiting",
            language === "ar" ? "MP3 320kbps + WAV 24-bit" : "MP3 320kbps + WAV 24-bit",
          ].map((item, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-zinc-500">
              <span className={i <= mixMsgIdx ? "text-violet-400" : "text-zinc-700"}>
                {i <= mixMsgIdx ? "✓" : "○"}
              </span>
              {item}
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Done ─────────────────────────────────────────────────────────────────
  if (step === "done") {
    return (
      <div className="flex flex-col items-center gap-6 w-full max-w-lg text-center" dir={isRTL ? "rtl" : "ltr"}>
        {/* Header */}
        <div className="space-y-3">
          <div className="h-20 w-20 mx-auto rounded-full bg-gradient-to-br from-violet-500/30 to-fuchsia-500/30 flex items-center justify-center text-4xl">
            🎵
          </div>
          <h2 className="text-2xl font-bold text-white">
            {language === "ar" ? "أغنيتك جاهزة" : "Your Song is Ready"}
          </h2>
          <p className="text-zinc-400 text-sm">
            {language === "ar"
              ? `"${projectName}" — مُنتج احترافياً بصوتك وإيقاعك`
              : `"${projectName}" — professionally produced with your voice`}
          </p>
        </div>

        {/* Player */}
        <div className="w-full rounded-xl border border-violet-500/20 bg-zinc-900 p-5 space-y-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-violet-500/20 flex items-center justify-center text-xl shrink-0">🎶</div>
            <div className="text-left">
              <p className="font-semibold text-white text-sm">{projectName}</p>
              <p className="text-xs text-zinc-500">Stereo • 48kHz • Mixed & Mastered</p>
            </div>
          </div>
          <audio controls src={mp3Url} className="w-full h-10" />
        </div>

        {/* Download buttons */}
        <div className="w-full grid grid-cols-2 gap-3">
          <a
            href={mp3Url}
            download={`${projectName}.mp3`}
            className="flex flex-col items-center gap-1 rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-4 hover:border-violet-500 hover:bg-violet-500/10 transition-all"
          >
            <span className="text-2xl">⬇️</span>
            <span className="text-sm font-semibold text-white">MP3</span>
            <span className="text-xs text-zinc-500">320kbps</span>
          </a>
          {wavUrl && (
            <a
              href={wavUrl}
              download={`${projectName}.wav`}
              className="flex flex-col items-center gap-1 rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-4 hover:border-violet-500 hover:bg-violet-500/10 transition-all"
            >
              <span className="text-2xl">⬇️</span>
              <span className="text-sm font-semibold text-white">WAV</span>
              <span className="text-xs text-zinc-500">24-bit / 48kHz</span>
            </a>
          )}
        </div>

        <div className="w-full rounded-xl border border-zinc-800 bg-zinc-900 p-4 space-y-3 text-left">
          <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            {language === "ar" ? "تقييم سريع" : "Quick rating"}
          </p>
          {[
            ["beat_quality", language === "ar" ? "الإيقاع" : "Beat quality"],
            ["vocal_preservation", language === "ar" ? "حفظ الصوت" : "Vocal preservation"],
            ["overall_satisfaction", language === "ar" ? "الرضا العام" : "Overall"],
          ].map(([key, label]) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <span className="text-sm text-zinc-300">{label}</span>
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setFeedback((f) => ({ ...f, [key]: value }))}
                    className={`h-8 w-8 rounded-lg border text-xs transition-colors ${
                      feedback[key as keyof typeof feedback] === value
                        ? "border-violet-500 bg-violet-500/20 text-violet-200"
                        : "border-zinc-700 text-zinc-500 hover:border-zinc-500"
                    }`}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <Button fullWidth variant="secondary" onClick={sendFeedback} disabled={feedbackSent}>
            {feedbackSent
              ? (language === "ar" ? "تم الحفظ" : "Saved")
              : (language === "ar" ? "إرسال" : "Send")}
          </Button>
        </div>

        {/* Actions */}
        <div className="w-full flex flex-col gap-3">
          <Button fullWidth onClick={() => router.push("/dashboard")}>
            {language === "ar" ? "العودة إلى أغانيّ" : "Back to My Songs"}
          </Button>
          <button
            className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
            onClick={() => router.push("/studio/new")}
          >
            {language === "ar" ? "+ إنشاء أغنية جديدة" : "+ Create another song"}
          </button>
        </div>
      </div>
    );
  }

  // ── Error ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col items-center gap-6 max-w-sm text-center" dir={isRTL ? "rtl" : "ltr"}>
      <div className="h-16 w-16 rounded-full bg-red-500/20 flex items-center justify-center text-3xl">⚠️</div>
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-white">{language === "ar" ? "فشل المزج" : "Mix Failed"}</h2>
        <p className="text-sm text-zinc-400">{errorMsg}</p>
      </div>
      <div className="flex gap-3 w-full">
        <Button variant="secondary" className="flex-1" onClick={() => router.push("/dashboard")}>
          {language === "ar" ? "العودة" : "Go Back"}
        </Button>
        <Button className="flex-1" onClick={runMix}>
          {language === "ar" ? "حاول مجدداً" : "Try Again"}
        </Button>
      </div>
    </div>
  );
}
