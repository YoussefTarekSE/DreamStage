"use client";

import { useEffect, useState } from "react";
import { useLanguage } from "@/hooks/useLanguage";

const STEPS_EN = [
  "Cleaning your recording...",
  "Leveling your vocal...",
  "Correcting pitch...",
  "Applying autotune...",
  "Finishing up...",
];
const STEPS_AR = [
  "تنظيف تسجيلك...",
  "ضبط مستوى صوتك...",
  "تصحيح الطبقة الصوتية...",
  "تطبيق الأوتوتيون...",
  "اللمسات الأخيرة...",
];

export function ProcessingStatus() {
  const { language } = useLanguage();
  const steps = language === "ar" ? STEPS_AR : STEPS_EN;
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, steps.length - 1));
    }, 4000);
    return () => clearInterval(interval);
  }, [steps.length]);

  return (
    <div className="flex flex-col items-center gap-6 py-8 text-center">
      <div className="relative h-20 w-20">
        <div className="absolute inset-0 rounded-full border-2 border-violet-500/30" />
        <div className="absolute inset-0 rounded-full border-2 border-t-violet-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center text-2xl">🎙️</div>
      </div>
      <div className="space-y-2">
        <p className="text-lg font-semibold text-white">
          {language === "ar" ? "يعالج الذكاء الاصطناعي صوتك" : "AI is processing your vocal"}
        </p>
        <p className="text-sm text-violet-400 animate-pulse min-h-[1.5rem]">
          {steps[stepIndex]}
        </p>
      </div>
      <p className="text-xs text-zinc-600">
        {language === "ar" ? "قد يستغرق هذا حتى 30 ثانية" : "This may take up to 30 seconds"}
      </p>
    </div>
  );
}
