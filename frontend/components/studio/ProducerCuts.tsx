"use client";

import { useState, useEffect, useCallback } from "react";
import { useLanguage } from "@/hooks/useLanguage";
import { listCuts, favoriteCut, restoreCut, type ProducerCut } from "@/lib/api";

interface ProducerCutsProps {
  projectId: string;
  /** Bumped by the parent after each generation so the list reloads. */
  refreshKey: number;
  /** Branch a new exploration from this cut. */
  onBranch: (cut: number) => void;
  /** Called after a cut is made current, so the parent can swap its player. */
  onRestored?: (cut: ProducerCut) => void;
}

/**
 * The Producer Cuts history — every cut ever made for this project. The artist
 * can replay, compare, favorite (★), branch from, and restore any past cut.
 * Nothing good is ever lost.
 */
export function ProducerCuts({ projectId, refreshKey, onBranch, onRestored }: ProducerCutsProps) {
  const { language, isRTL } = useLanguage();
  const [cuts, setCuts] = useState<ProducerCut[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<number | null>(null);
  const [restoredCut, setRestoredCut] = useState<number | null>(null);
  const [actionError, setActionError] = useState("");

  const load = useCallback(async () => {
    try {
      const data = await listCuts(projectId);
      setCuts(data.cuts);
    } catch {
      setCuts([]);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void load();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [load, refreshKey]);

  async function toggleFav(cut: number) {
    setBusy(cut);
    setActionError("");
    try {
      const { favorite } = await favoriteCut(projectId, cut);
      setCuts((cs) => cs.map((c) => (c.cut === cut ? { ...c, favorite } : c)));
    } catch {
      // The star is a taste-learning signal — never let it fail silently.
      setActionError(language === "ar"
        ? "تعذّر حفظ التفضيل — حاول مرة أخرى."
        : "Couldn't save the favorite — try again.");
    } finally {
      setBusy(null);
    }
  }

  async function restore(cut: number) {
    setBusy(cut);
    setActionError("");
    try {
      await restoreCut(projectId, cut);
      setRestoredCut(cut);
      const restored = cuts.find((c) => c.cut === cut);
      if (restored) onRestored?.(restored);
    } catch {
      setActionError(language === "ar"
        ? "تعذّرت الاستعادة — حاول مرة أخرى."
        : "Couldn't make this cut current — try again.");
    } finally {
      setBusy(null);
    }
  }

  if (loading) return null;
  if (cuts.length <= 1) return null; // nothing to compare yet

  // newest first
  const ordered = [...cuts].sort((a, b) => b.cut - a.cut);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 space-y-3" dir={isRTL ? "rtl" : "ltr"}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          {language === "ar" ? "نسخ المنتِج" : "Producer Cuts"}
        </h3>
        <span className="text-xs text-zinc-500">
          {cuts.length} {language === "ar" ? "نسخة" : (cuts.length === 1 ? "cut" : "cuts")}
        </span>
      </div>

      {actionError && (
        <p className="text-xs text-red-400">{actionError}</p>
      )}

      <div className="flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
        {ordered.map((c) => (
          <div key={c.cut} className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 space-y-2">
            <div className="flex items-center gap-2">
              <button
                onClick={() => toggleFav(c.cut)}
                disabled={busy === c.cut}
                aria-label="favorite"
                className={`text-lg leading-none transition-colors ${c.favorite ? "text-amber-400" : "text-zinc-600 hover:text-zinc-300"}`}
              >
                {c.favorite ? "★" : "☆"}
              </button>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {c.label}
                  {c.parent_cut != null && (
                    <span className="ml-2 text-xs text-violet-400/80">
                      {language === "ar" ? `تفرّع من ${c.parent_cut}` : `branch of ${c.parent_cut}`}
                    </span>
                  )}
                </p>
                <p className="text-xs text-zinc-500 truncate">
                  {c.genre_label || c.genre} • {c.key} • {c.tempo} BPM
                </p>
              </div>
              <span className="text-xs text-violet-400/90 border border-violet-500/30 rounded-full px-2 py-0.5 capitalize shrink-0">
                {c.emotion}
              </span>
            </div>

            {(language === "ar" ? c.note_ar : c.note_en) && (
              <p className="text-xs text-zinc-400 leading-relaxed">
                {language === "ar" ? c.note_ar : c.note_en}
              </p>
            )}

            {c.beat_url && <audio controls src={c.beat_url} className="w-full h-8" />}

            <div className="flex gap-2">
              <button
                onClick={() => onBranch(c.cut)}
                className="flex-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-violet-500 hover:text-violet-300 transition-colors"
              >
                {language === "ar" ? "تفرّع من هنا" : "Branch from this"}
              </button>
              <button
                onClick={() => restore(c.cut)}
                disabled={busy === c.cut}
                className={`flex-1 rounded-md border px-2 py-1 text-xs transition-colors ${
                  restoredCut === c.cut
                    ? "border-violet-500 text-violet-300"
                    : "border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-white"
                }`}
              >
                {restoredCut === c.cut
                  ? (language === "ar" ? "النسخة الحالية ✓" : "Current ✓")
                  : (language === "ar" ? "استعادة" : "Make current")}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
