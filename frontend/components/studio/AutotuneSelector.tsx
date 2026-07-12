"use client";

import { useLanguage } from "@/hooks/useLanguage";

export type AutotuneLevel =
  | "natural" | "subtle" | "modern_pop" | "rnb"
  | "rap" | "melodic" | "heavy" | "none";

const LEVELS: { value: AutotuneLevel; label_en: string; label_ar: string; desc_en: string; desc_ar: string }[] = [
  { value: "natural",    label_en: "Natural",     label_ar: "طبيعي",      desc_en: "Cleanup only, no pitch change",      desc_ar: "تنظيف فقط، بدون تغيير الطبقة" },
  { value: "subtle",     label_en: "Subtle",      label_ar: "خفيف",       desc_en: "Light correction, sounds like you",  desc_ar: "تصحيح خفيف، يبدو كأنك أنت" },
  { value: "modern_pop", label_en: "Modern Pop",  label_ar: "بوب عصري",   desc_en: "Polished & bright, pop production",  desc_ar: "صوت مصقول ومشرق بإنتاج بوب" },
  { value: "rnb",        label_en: "R&B",         label_ar: "آر أند بي",  desc_en: "Warm & smooth, soulful tone",        desc_ar: "دافئ وسلس بنبرة روحانية" },
  { value: "rap",        label_en: "Rap",         label_ar: "راب",        desc_en: "Present & punchy, beat-driven",      desc_ar: "حاضر وقوي، يقوده الإيقاع" },
  { value: "melodic",    label_en: "Melodic",     label_ar: "لحني",       desc_en: "Tuned & wide, melodic singing",      desc_ar: "مضبوط وواسع للغناء اللحني" },
  { value: "heavy",      label_en: "Heavy",       label_ar: "قوي",        desc_en: "Strong effect, T-Pain style",        desc_ar: "تأثير قوي بأسلوب T-Pain" },
  { value: "none",       label_en: "No Autotune", label_ar: "بدون أوتوتيون", desc_en: "Clean & natural, zero tuning",    desc_ar: "نظيف وطبيعي بدون ضبط" },
];

interface AutotuneSelectorProps {
  value: AutotuneLevel;
  onChange: (level: AutotuneLevel) => void;
}

export function AutotuneSelector({ value, onChange }: AutotuneSelectorProps) {
  const { language, isRTL } = useLanguage();

  return (
    <div className="w-full space-y-3" dir={isRTL ? "rtl" : "ltr"}>
      <p className="text-sm font-medium text-zinc-300">
        {language === "ar" ? "النمط الصوتي" : "Vocal Style"}
      </p>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {LEVELS.map((level) => {
          const selected = value === level.value;
          return (
            <button
              key={level.value}
              onClick={() => onChange(level.value)}
              className={`flex flex-col items-center gap-1 rounded-lg border px-3 py-3 text-center transition-all ${
                selected
                  ? "border-violet-500 bg-violet-500/15 text-violet-300"
                  : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200"
              }`}
            >
              <span className="text-sm font-semibold">
                {language === "ar" ? level.label_ar : level.label_en}
              </span>
              <span className="text-xs opacity-70 leading-tight">
                {language === "ar" ? level.desc_ar : level.desc_en}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
