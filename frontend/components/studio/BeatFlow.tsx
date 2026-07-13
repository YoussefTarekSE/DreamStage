"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/hooks/useLanguage";
import { generateBeat, acceptBeat, getBeatUrl, getActiveBeatJob, pollBeatJob, type BeatResult } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ProducerCuts } from "@/components/studio/ProducerCuts";

type Step = "loading_existing" | "empty" | "generating" | "preview" | "style_input" | "error";

// DreamStage is an AI producer: the session is UNLIMITED. Every generation is a
// Producer Cut, kept forever — the artist decides when the song is finished.

interface BeatMeta {
  genre: string;
  key: string;
  tempo_bpm: number;
  emotion: string;
  message_en: string;
  message_ar: string;
}

interface BeatFlowProps {
  projectId: string;
  existingBeatUrl?: string;
  hasExistingBeat?: boolean;
  existingAttempts?: number;
}

const STYLE_PICKS = [
  { en: "Hip Hop Trap",  ar: "هيب هوب تراب" },
  { en: "R&B Soul",      ar: "R&B soul" },
  { en: "Afrobeats",     ar: "أفروبيتس" },
  { en: "Phonk",         ar: "فونك" },
  { en: "Amapiano",      ar: "أماپيانو" },
  { en: "UK Drill",      ar: "دريل بريطاني" },
  { en: "Dancehall",     ar: "دانسهول" },
  { en: "Chill Lo-Fi",   ar: "تشيل لو-فاي" },
];

export function BeatFlow({ projectId, existingBeatUrl, hasExistingBeat = false, existingAttempts = 0 }: BeatFlowProps) {
  const router = useRouter();
  const { language, isRTL } = useLanguage();

  const [step, setStep]         = useState<Step>(existingBeatUrl ? "preview" : hasExistingBeat ? "loading_existing" : "empty");
  const [beatUrl, setBeatUrl]   = useState(existingBeatUrl ?? "");
  const [attempts, setAttempts] = useState(existingAttempts);
  const [cutLabel, setCutLabel] = useState("");
  const [meta, setMeta]         = useState<BeatMeta | null>(null);
  const [styleHint, setStyleHint] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [accepting, setAccepting] = useState(false);
  const [acceptError, setAcceptError] = useState("");
  const [genElapsed, setGenElapsed] = useState(0);
  const [prevStep, setPrevStep] = useState(step);

  // Reset the counter during render on entry into "generating" (adjusting
  // state in response to a change, rather than resetting it from the effect).
  if (step !== prevStep) {
    setPrevStep(step);
    if (step === "generating") setGenElapsed(0);
  }

  // Real elapsed-time counter for the generating screen (honest progress —
  // neural cuts take 1-2 minutes and the copy is staged off actual time).
  useEffect(() => {
    if (step !== "generating") return;
    const t0 = Date.now();
    const timer = window.setInterval(
      () => setGenElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [step]);

  async function loadExistingBeat() {
    try {
      const result = await getBeatUrl(projectId);
      setBeatUrl(result.beat_url);
      setAttempts(result.beat_attempts ?? existingAttempts);
      setMeta(result.last_genre ? {
        genre: result.last_genre,
        key: "",
        tempo_bpm: 0,
        emotion: "",
        message_en: "",
        message_ar: "",
      } : null);
      setStep("preview");
    } catch {
      setErrorMsg(language === "ar" ? "لم نتمكن من تحميل الإيقاع." : "Could not load the existing beat.");
      setStep("error");
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      // A generation may still be running from before a page refresh —
      // re-attach to it instead of losing the cut (jobs survive the client).
      try {
        const jobId = await getActiveBeatJob(projectId);
        if (jobId) {
          setStep("generating");
          try {
            applyResult(await pollBeatJob(projectId, jobId));
          } catch (err) {
            showGenerateError(err);
          }
          return;
        }
      } catch {
        /* resume is best-effort */
      }
      if (!existingBeatUrl && hasExistingBeat) void loadExistingBeat();
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function applyResult(result: BeatResult) {
    setBeatUrl(result.beat_url);
    setAttempts(result.total_cuts ?? result.attempt);
    setCutLabel(result.cut_label ?? `Cut ${result.attempt}`);
    setMeta({
      genre:      result.genre,
      key:        result.key,
      tempo_bpm:  result.tempo_bpm,
      emotion:    result.emotion,
      message_en: result.message_en,
      message_ar: result.message_ar,
    });
    setStep("preview");
  }

  function showGenerateError(err: unknown) {
    const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
    const detail = e?.detail;
    if (detail && typeof detail === "object" && detail.message_en) {
      setErrorMsg(language === "ar" ? (detail.message_ar ?? detail.message_en) : detail.message_en);
    } else if (typeof detail === "string") {
      setErrorMsg(detail);
    } else {
      setErrorMsg(language === "ar" ? "فشل توليد الإيقاع. يرجى المحاولة مرة أخرى." : "Beat generation failed. Please try again.");
    }
    setStep("error");
  }

  async function runGenerate(hint: string, branchFrom: number | null = null) {
    setStep("generating");
    try {
      applyResult(await generateBeat(projectId, hint, branchFrom));
    } catch (err: unknown) {
      showGenerateError(err);
    }
  }

  async function handleAccept() {
    setAccepting(true);
    setAcceptError("");
    try {
      await acceptBeat(projectId);
      router.push(`/studio/${projectId}/coach`);
    } catch (err: unknown) {
      const e = err as { detail?: { message_en?: string; message_ar?: string } | string };
      const detail = e?.detail;
      if (detail && typeof detail === "object" && detail.message_en) {
        setAcceptError(language === "ar" ? (detail.message_ar ?? detail.message_en) : detail.message_en);
      } else {
        setAcceptError(language === "ar"
          ? "تعذّر حفظ اختيارك — تحقق من اتصالك وحاول مرة أخرى."
          : "Couldn't save your choice — check your connection and try again.");
      }
      setAccepting(false);
    }
  }

  // ── Generating ─────────────────────────────────────────────────────────────
  if (step === "loading_existing") {
    return (
      <div className="flex flex-col items-center gap-4 text-center py-8">
        <div className="h-10 w-10 rounded-full border-2 border-t-violet-500 border-zinc-800 animate-spin" />
        <p className="text-sm text-zinc-400">
          {language === "ar" ? "جارٍ تحميل الإيقاع..." : "Loading beat..."}
        </p>
      </div>
    );
  }

  if (step === "empty") {
    return (
      <div className="flex flex-col gap-6 w-full max-w-md text-center" dir={isRTL ? "rtl" : "ltr"}>
        <div>
          <h2 className="text-xl font-bold text-white">
            {language === "ar" ? "جاهز لصنع الإيقاع؟" : "Ready to make the beat?"}
          </h2>
          <p className="text-sm text-zinc-400 mt-1">
            {language === "ar" ? "ابدأ عندما تكون جاهزاً." : "Start when you are ready."}
          </p>
        </div>
        <div className="flex gap-3">
          <Button className="flex-1" onClick={() => runGenerate("")}>
            {language === "ar" ? "توليد" : "Generate"}
          </Button>
          <Button variant="secondary" className="flex-1" onClick={() => setStep("style_input")}>
            {language === "ar" ? "أسلوب" : "Style"}
          </Button>
        </div>
      </div>
    );
  }

  if (step === "generating") {
    return (
      <div className="flex flex-col items-center gap-6 text-center py-8">
        <div className="relative h-24 w-24">
          <div className="absolute inset-0 rounded-full border-2 border-violet-500/20" />
          <div className="absolute inset-0 rounded-full border-2 border-t-violet-500 animate-spin" />
          <div
            className="absolute inset-2 rounded-full border-2 border-t-violet-300/50 animate-spin"
            style={{ animationDirection: "reverse", animationDuration: "1.5s" }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-3xl">🎵</div>
        </div>
        <div className="space-y-2">
          <p className="text-xl font-bold text-white">
            {language === "ar" ? "توليد الإيقاع" : "Generating Your Beat"}
          </p>
          <p className="text-sm text-zinc-400 max-w-sm min-h-[2.5rem]">
            {(() => {
              // Honest staged copy keyed to real elapsed time — a neural cut
              // takes 1-2 minutes; never promise less than reality.
              const s = genElapsed;
              if (language === "ar") {
                if (s < 12) return "الاستماع إلى أدائك — المفتاح والإيقاع والطاقة...";
                if (s < 40) return "الفرقة العصبية تؤلّف حول أدائك...";
                if (s < 95) return "تسجيل الدرامز والباص والمفاتيح...";
                return "اللمسات الأخيرة وحفظ النسخة...";
              }
              if (s < 12) return "Listening to your take — key, tempo, energy...";
              if (s < 40) return "The neural band is composing around your performance...";
              if (s < 95) return "Recording drums, bass and keys...";
              return "Finishing touches and saving your cut...";
            })()}
          </p>
        </div>
        <div className="space-y-1 text-xs text-zinc-600">
          <p>
            {language === "ar"
              ? `عادةً ١-٢ دقيقة • ${genElapsed} ثانية`
              : `Usually 1-2 minutes • ${genElapsed}s elapsed`}
          </p>
          <p>{language === "ar" ? "كل إيقاع مصنوع خصيصاً لصوتك" : "Every beat is crafted specifically for your voice"}</p>
        </div>
      </div>
    );
  }

  // ── Style input ────────────────────────────────────────────────────────────
  if (step === "style_input") {
    return (
      <div className="flex flex-col gap-6 w-full max-w-md" dir={isRTL ? "rtl" : "ltr"}>
        <div>
          <h2 className="text-xl font-bold text-white mb-1">
            {language === "ar" ? "أخبرني عن الأسلوب الذي تريده" : "Tell me the style you want"}
          </h2>
          <p className="text-sm text-zinc-400">
            {language === "ar" ? "نسخة منتِج جديدة" : "A new Producer Cut"}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {STYLE_PICKS.map((s) => {
            const label = language === "ar" ? s.ar : s.en;
            return (
              <button
                key={s.en}
                onClick={() => setStyleHint(label)}
                className={`rounded-lg border px-3 py-2 text-sm transition-colors text-left ${
                  styleHint === label
                    ? "border-violet-500 bg-violet-500/15 text-violet-300"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-white"
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>

        <Input
          placeholder={language === "ar" ? "أو اكتب أسلوبك الخاص..." : "Or describe your own style..."}
          value={styleHint}
          onChange={(e) => setStyleHint(e.target.value)}
          maxLength={120}
        />

        <div className="flex gap-3">
          <Button variant="secondary" className="flex-1" onClick={() => setStep(beatUrl ? "preview" : "empty")}>
            {language === "ar" ? "إلغاء" : "Cancel"}
          </Button>
          <Button className="flex-1" onClick={() => runGenerate(styleHint)}>
            {language === "ar" ? "توليد" : "Generate"}
          </Button>
        </div>
      </div>
    );
  }

  // ── Preview ────────────────────────────────────────────────────────────────
  if (step === "preview") {
    const vibeMsg = meta ? (language === "ar" ? meta.message_ar : meta.message_en) : "";

    return (
      <div className="flex flex-col gap-6 w-full max-w-lg" dir={isRTL ? "rtl" : "ltr"}>
        <div>
          <h2 className="text-xl font-bold text-white">
            {cutLabel || (language === "ar" ? "نسختك جاهزة" : "Your Cut is Ready")}
          </h2>
          <p className="text-sm text-zinc-400 mt-1">
            {language === "ar"
              ? `${attempts} نسخة منتِج حتى الآن — استمر في الاستكشاف`
              : `${attempts} producer cut${attempts === 1 ? "" : "s"} so far — keep exploring`}
          </p>
        </div>

        {/* Beat metadata card */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-5 space-y-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-violet-500/20 flex items-center justify-center text-xl">🥁</div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-white text-sm">
                {meta?.genre ?? (language === "ar" ? "إيقاع مخصص" : "Custom Beat")}
              </p>
              {meta && (
                <p className="text-xs text-zinc-500">
                  {meta.key} • {meta.tempo_bpm} BPM
                </p>
              )}
            </div>
            {meta && (
              <span className="text-xs text-violet-400 border border-violet-500/30 rounded-full px-2 py-0.5 capitalize">
                {meta.emotion}
              </span>
            )}
          </div>

          {vibeMsg && (
            <p className="text-xs text-zinc-400 italic border-t border-zinc-800 pt-3">
              {vibeMsg}
            </p>
          )}

          <audio controls src={beatUrl} className="w-full h-10" />
        </div>

        {/* Producer-cut history dots — one per cut made this session (unlimited) */}
        {attempts > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {Array.from({ length: attempts }).map((_, i) => (
              <div
                key={i}
                className={`h-1.5 w-6 rounded-full ${i === attempts - 1 ? "bg-violet-400" : "bg-violet-500/40"}`}
              />
            ))}
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-col gap-3">
          {acceptError && (
            <p className="text-sm text-red-400 text-center">{acceptError}</p>
          )}
          <Button fullWidth onClick={handleAccept} loading={accepting}>
            {language === "ar" ? "أقبل هذا الإيقاع ✓" : "Accept This Beat ✓"}
          </Button>

          <div className="flex gap-3">
            <Button
              variant="secondary"
              className="flex-1"
              onClick={() => runGenerate("")}
            >
              {language === "ar" ? "نسخة أخرى" : "Another Cut"}
            </Button>
            <Button
              variant="secondary"
              className="flex-1"
              onClick={() => setStep("style_input")}
            >
              {language === "ar" ? "وجِّه المنتِج" : "Direct the Producer"}
            </Button>
          </div>

          <p className="text-center text-xs text-zinc-600">
            {language === "ar"
              ? "أنشئ ما تشاء من النسخ — لن تُفقد أي فكرة جيدة."
              : "Create as many cuts as you like — no good idea is ever lost."}
          </p>
        </div>

        {/* Full Producer Cuts history: replay, favorite, branch, restore */}
        <ProducerCuts
          projectId={projectId}
          refreshKey={attempts}
          onBranch={(cut) => runGenerate("", cut)}
          onRestored={(cut) => {
            // Keep the main player in sync with the newly-current cut so the
            // artist always accepts exactly what they are hearing.
            if (cut.beat_url) setBeatUrl(cut.beat_url);
            setCutLabel(cut.label);
            setMeta((m) => ({
              genre:      cut.genre,
              key:        cut.key,
              tempo_bpm:  cut.tempo,
              emotion:    cut.emotion,
              message_en: m?.message_en ?? "",
              message_ar: m?.message_ar ?? "",
            }));
          }}
        />
      </div>
    );
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-sm text-center" dir={isRTL ? "rtl" : "ltr"}>
      <div className="h-16 w-16 rounded-full bg-red-500/20 flex items-center justify-center text-3xl">⚠️</div>
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-white">
          {language === "ar" ? "فشل توليد الإيقاع" : "Beat generation failed"}
        </h2>
        <p className="text-sm text-zinc-400">{errorMsg}</p>
      </div>
      <div className="flex gap-3 w-full">
        <Button variant="secondary" className="flex-1" onClick={() => router.push("/dashboard")}>
          {language === "ar" ? "العودة" : "Go Back"}
        </Button>
        <Button className="flex-1" onClick={() => runGenerate("")}>
          {language === "ar" ? "حاول مجدداً" : "Try Again"}
        </Button>
      </div>
    </div>
  );
}
