"use client";

import { useRef, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/hooks/useLanguage";
import { getCoachFeedback, skipCoaching, replaceVocal, type CoachFeedback, type CoachSection } from "@/lib/api";
import { Button } from "@/components/ui/Button";

type Step = "loading" | "feedback" | "error";

interface CoachFlowProps {
  projectId: string;
  cachedFeedback?: CoachFeedback;
}

export function CoachFlow({ projectId, cachedFeedback }: CoachFlowProps) {
  const router = useRouter();
  const { language, isRTL } = useLanguage();

  const [step, setStep] = useState<Step>(cachedFeedback ? "feedback" : "loading");
  const [feedback, setFeedback] = useState<CoachFeedback | null>(cachedFeedback ?? null);
  const [errorMsg, setErrorMsg] = useState("");
  const [skipping, setSkipping] = useState(false);
  const [continueError, setContinueError] = useState("");
  const [replacing, setReplacing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!cachedFeedback) {
      load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function load() {
    try {
      const data = await getCoachFeedback(projectId);
      setFeedback(data);
      setStep("feedback");
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
      const d = e?.detail;
      if (d && typeof d === "object" && d.message_en) {
        setErrorMsg(language === "ar" ? (d.message_ar ?? d.message_en) : d.message_en);
      } else {
        setErrorMsg(language === "ar" ? "تعذّر تحميل ملاحظات المدرّب." : "Could not load coach feedback.");
      }
      setStep("error");
    }
  }

  async function handleContinue() {
    setSkipping(true);
    setContinueError("");
    try {
      await skipCoaching(projectId);
      router.push(`/studio/${projectId}/mix`);
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
      const d = e?.detail;
      if (d && typeof d === "object" && d.message_en) {
        setContinueError(language === "ar" ? (d.message_ar ?? d.message_en) : d.message_en);
      } else {
        setContinueError(language === "ar"
          ? "تعذّر المتابعة — تحقق من اتصالك وحاول مرة أخرى."
          : "Couldn't continue — check your connection and try again.");
      }
      setSkipping(false);
    }
  }

  // ── Loading ───────────────────────────────────────────────────────────────
  async function handleReplaceVocal(file: File | undefined) {
    if (!file) return;
    setReplacing(true);
    try {
      await replaceVocal(projectId, file);
      router.push(`/studio/${projectId}/beat`);
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
      const d = e?.detail;
      setErrorMsg(d && typeof d === "object" && d.message_en ? (language === "ar" ? (d.message_ar ?? d.message_en) : d.message_en) : "Could not replace vocal.");
      setStep("error");
    } finally {
      setReplacing(false);
    }
  }

  if (step === "loading") {
    const msgs = language === "ar"
      ? ["يستمع إلى أدائك...", "يحلّل الطبقات الصوتية...", "يكتب ملاحظاته...", "يختار الكلمات المناسبة..."]
      : ["Listening to your performance...", "Analysing pitch and dynamics...", "Writing personalised feedback...", "Choosing the right words..."];
    return <CoachLoader messages={msgs} />;
  }

  // ── Error ─────────────────────────────────────────────────────────────────
  if (step === "error" || !feedback) {
    return (
      <div className="flex flex-col items-center gap-6 text-center max-w-sm" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-16 w-16 rounded-full bg-red-500/20 flex items-center justify-center text-3xl">⚠️</div>
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-white">{language === "ar" ? "خطأ" : "Error"}</h2>
          <p className="text-sm text-zinc-400">{errorMsg}</p>
        </div>
        <div className="flex gap-3 w-full">
          <Button variant="secondary" className="flex-1" onClick={() => router.push("/dashboard")}>
            {language === "ar" ? "العودة" : "Go Back"}
          </Button>
          <Button className="flex-1" onClick={() => { setStep("loading"); load(); }}>
            {language === "ar" ? "حاول مجدداً" : "Try Again"}
          </Button>
        </div>
      </div>
    );
  }

  // ── Feedback ──────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-6 w-full max-w-lg" dir={isRTL ? "rtl" : "ltr"}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-xl bg-violet-500/20 flex items-center justify-center text-2xl">🎙️</div>
        <div>
          <h2 className="text-xl font-bold text-white">
            {language === "ar" ? "ملاحظات المدرّب" : "Producer's Notes"}
          </h2>
          <div className="flex gap-1 mt-1">
            {Array.from({ length: 5 }).map((_, i) => (
              <span key={i} className={`text-sm ${i < (feedback.rating ?? 4) ? "text-yellow-400" : "text-zinc-700"}`}>★</span>
            ))}
          </div>
        </div>
      </div>

      {/* Overall assessment */}
      <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 p-4">
        <p className="text-sm text-zinc-200 leading-relaxed">{feedback.overall_assessment}</p>
      </div>

      {/* Strengths */}
      {feedback.strengths?.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
            {language === "ar" ? "نقاط القوة" : "What's Working"}
          </p>
          <div className="space-y-2">
            {feedback.strengths.map((s, i) => (
              <div key={i} className="flex items-start gap-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2.5">
                <span className="text-emerald-400 mt-0.5">✓</span>
                <p className="text-sm text-zinc-200">{s}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sections */}
      {feedback.sections?.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
            {language === "ar" ? "تفاصيل الأداء" : "Section by Section"}
          </p>
          <div className="space-y-3">
            {feedback.sections.map((section) => (
              <SectionCard key={section.id} section={section} language={language} />
            ))}
          </div>
        </div>
      )}

      {/* Final message */}
      {feedback.final_message && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-center">
          <p className="text-sm text-zinc-300 italic">&quot;{feedback.final_message}&quot;</p>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-col gap-3 pt-2">
        {continueError && (
          <p className="text-sm text-red-400 text-center">{continueError}</p>
        )}
        <Button fullWidth onClick={handleContinue} loading={skipping}>
          {language === "ar" ? "المتابعة إلى الميكس →" : "Continue to Mix →"}
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(event) => void handleReplaceVocal(event.target.files?.[0])}
        />
        <button
          className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors text-center"
          disabled={replacing}
          onClick={() => fileInputRef.current?.click()}
        >
          {replacing ? "Replacing..." : "Replace vocal"}
        </button>
      </div>
    </div>
  );
}

function SectionCard({ section, language }: { section: CoachSection; language: string }) {
  const colors = {
    strength:    "border-emerald-500/20 bg-emerald-500/5",
    improvement: "border-amber-500/20 bg-amber-500/5",
    critical:    "border-red-500/20 bg-red-500/5",
  };
  const icons = { strength: "✓", improvement: "◎", critical: "⚠" };
  const iconColors = { strength: "text-emerald-400", improvement: "text-amber-400", critical: "text-red-400" };

  const type = section.type ?? "improvement";

  return (
    <div className={`rounded-lg border p-4 space-y-2 ${colors[type]}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <span className={`mt-0.5 text-sm font-bold ${iconColors[type]}`}>{icons[type]}</span>
          <p className="text-sm text-zinc-200">{section.observation}</p>
        </div>
        <span className="text-xs text-zinc-600 whitespace-nowrap shrink-0">{section.time_hint}</span>
      </div>
      {section.fix && (
        <p className="text-xs text-zinc-400 leading-relaxed pl-5">
          {language === "ar" ? "💡 " : "💡 "}{section.fix}
        </p>
      )}
      {section.should_rerecord && (
        <div className="pl-5">
          <span className="inline-flex items-center gap-1 text-xs text-amber-400 border border-amber-500/30 rounded-full px-2 py-0.5">
            {language === "ar" ? "يُنصح بإعادة التسجيل" : "Re-record recommended"}
          </span>
        </div>
      )}
    </div>
  );
}

function CoachLoader({ messages }: { messages: string[] }) {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setMsgIndex((i) => Math.min(i + 1, messages.length - 1)), 4000);
    return () => clearInterval(t);
  }, [messages.length]);

  return (
    <div className="flex flex-col items-center gap-6 py-8 text-center">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 rounded-full border-2 border-violet-500/20" />
        <div className="absolute inset-0 rounded-full border-2 border-t-violet-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center text-2xl">🎙️</div>
      </div>
      <div className="space-y-2">
        <p className="text-lg font-semibold text-white">Your AI Producer is listening</p>
        <p className="text-sm text-violet-400 animate-pulse min-h-[1.25rem]">{messages[msgIndex]}</p>
      </div>
    </div>
  );
}
