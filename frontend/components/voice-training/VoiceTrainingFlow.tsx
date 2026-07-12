"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useLanguage } from "@/hooks/useLanguage";
import { submitVoiceTraining } from "@/lib/api";
import { PhraseRecorder } from "./PhraseRecorder";
import { Button } from "@/components/ui/Button";

const PHRASES = {
  en: [
    "The stars are calling me home tonight",
    "I walk alone but I'm never lost",
    "Hold on, let the music set you free",
  ],
  ar: [
    "النجوم تناديني للعودة إلى البيت",
    "أمشي وحدي لكنني لست ضائعاً",
    "تمسّك بالموسيقى وستحرّرك",
  ],
};

type FlowStep = "intro" | "recording" | "submitting" | "done" | "error";

export function VoiceTrainingFlow() {
  const router = useRouter();
  const { language, isRTL } = useLanguage();
  const phrases = PHRASES[language as keyof typeof PHRASES] ?? PHRASES.en;

  const [step, setStep] = useState<FlowStep>("intro");
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [recordings, setRecordings] = useState<Blob[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [confirmationMsg, setConfirmationMsg] = useState("");

  function handlePhraseAccepted(blob: Blob) {
    const updated = [...recordings, blob];
    setRecordings(updated);

    if (phraseIndex < phrases.length - 1) {
      setPhraseIndex(phraseIndex + 1);
    } else {
      handleSubmit(updated);
    }
  }

  async function handleSubmit(blobs: Blob[]) {
    setStep("submitting");
    try {
      const result = await submitVoiceTraining(blobs, language);
      setConfirmationMsg(language === "ar" ? result.message_ar : result.message_en);
      setStep("done");
    } catch (err: unknown) {
      console.error("Voice training error:", JSON.stringify(err, null, 2));
      const e = err as { detail?: { message_en?: string; message_ar?: string; debug?: string } | string };
      const detail = e?.detail;
      if (detail && typeof detail === "object" && detail.message_en) {
        const msg = language === "ar" ? (detail.message_ar ?? detail.message_en) : detail.message_en;
        setErrorMsg(detail.debug ? `${msg}\n\nDebug: ${detail.debug}` : msg);
      } else if (typeof detail === "string") {
        setErrorMsg(detail);
      } else {
        setErrorMsg(language === "ar" ? "حدث خطأ. يرجى المحاولة مرة أخرى." : "Something went wrong. Please try again.");
      }
      setStep("error");
    }
  }

  if (step === "intro") {
    return (
      <div className="flex flex-col items-center gap-8 text-center max-w-md" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-20 w-20 rounded-full bg-violet-500/20 flex items-center justify-center text-4xl">
          🎤
        </div>
        <div className="space-y-3">
          <h1 className="text-2xl font-bold text-white">
            {language === "ar" ? "لنتعرف على صوتك" : "Let's learn your voice"}
          </h1>
          <p className="text-zinc-400 leading-relaxed">
            {language === "ar"
              ? "سأعرض عليك ٣ عبارات قصيرة لتغنّيها. هذا يساعدني على فهم نطاقك وجرسك وأسلوبك — حتى يبدو كل شيء نصنعه معاً كأنه أنت."
              : "I'll show you 3 short phrases to sing. This helps me understand your range, your tone, and your style — so everything we make together sounds like you."}
          </p>
          <p className="text-sm text-zinc-500">
            {language === "ar" ? "يستغرق حوالي دقيقتين" : "Takes about 2 minutes"}
          </p>
        </div>
        <div className="flex flex-col gap-2 text-sm text-zinc-500 w-full text-start rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <p className="font-medium text-zinc-300">
            {language === "ar" ? "نصائح للحصول على أفضل نتيجة:" : "Tips for best results:"}
          </p>
          <ul className="space-y-1 list-disc list-inside">
            <li>{language === "ar" ? "اجلس في مكان هادئ" : "Find a quiet spot"}</li>
            <li>{language === "ar" ? "غنّ بصوت طبيعي — لا تحاول التأثير" : "Sing naturally — don't try to perform"}</li>
            <li>{language === "ar" ? "الميكروفون على بُعد 20-30 سم من فمك" : "Mic 20–30cm from your mouth"}</li>
          </ul>
        </div>
        <Button fullWidth onClick={() => setStep("recording")}>
          {language === "ar" ? "ابدأ التدريب الصوتي" : "Start Voice Training"}
        </Button>
      </div>
    );
  }

  if (step === "recording") {
    return (
      <PhraseRecorder
        key={phraseIndex}
        phrase={phrases[phraseIndex]}
        phraseIndex={phraseIndex}
        totalPhrases={phrases.length}
        onAccept={handlePhraseAccepted}
        isRTL={isRTL}
      />
    );
  }

  if (step === "submitting") {
    return (
      <div className="flex flex-col items-center gap-6 text-center" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-16 w-16 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
        <div className="space-y-2">
          <p className="text-lg font-semibold text-white">
            {language === "ar" ? "أحلل صوتك..." : "Analysing your voice..."}
          </p>
          <p className="text-sm text-zinc-400">
            {language === "ar" ? "يستغرق هذا حوالي 30 ثانية" : "This takes about 30 seconds"}
          </p>
        </div>
        {["Cleaning recordings...", "Detecting your range...", "Building Voice Profile..."].map((msg, i) => (
          <p key={i} className="text-xs text-zinc-600 animate-pulse" style={{ animationDelay: `${i * 0.4}s` }}>
            {language === "ar"
              ? ["تنظيف التسجيلات...", "تحديد نطاقك...", "بناء ملفك الصوتي..."][i]
              : msg}
          </p>
        ))}
      </div>
    );
  }

  if (step === "done") {
    return (
      <div className="flex flex-col items-center gap-6 text-center max-w-sm" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-20 w-20 rounded-full bg-violet-500/20 flex items-center justify-center text-4xl">
          ✨
        </div>
        <div className="space-y-3">
          <h2 className="text-2xl font-bold text-white">
            {language === "ar" ? "ملفك الصوتي جاهز" : "Your Voice Profile is ready"}
          </h2>
          <p className="text-zinc-400 leading-relaxed">{confirmationMsg}</p>
        </div>
        <Button fullWidth onClick={() => router.push("/dashboard")}>
          {language === "ar" ? "لنصنع شيئاً" : "Let's make something"}
        </Button>
      </div>
    );
  }

  if (step === "error") {
    return (
      <div className="flex flex-col items-center gap-6 text-center max-w-sm" dir={isRTL ? "rtl" : "ltr"}>
        <div className="h-20 w-20 rounded-full bg-red-500/20 flex items-center justify-center text-4xl">
          ⚠️
        </div>
        <div className="space-y-3">
          <h2 className="text-xl font-bold text-white">
            {language === "ar" ? "مشكلة في التسجيل" : "Recording issue"}
          </h2>
          <p className="text-zinc-400">{errorMsg}</p>
        </div>
        <div className="flex gap-3 w-full">
          <Button variant="secondary" className="flex-1" onClick={() => {
            setStep("recording");
            setPhraseIndex(0);
            setRecordings([]);
          }}>
            {language === "ar" ? "المحاولة مجدداً" : "Try Again"}
          </Button>
          <Button className="flex-1" onClick={() => router.push("/dashboard")}>
            {language === "ar" ? "تخطي الآن" : "Skip for Now"}
          </Button>
        </div>
      </div>
    );
  }

  return null;
}
