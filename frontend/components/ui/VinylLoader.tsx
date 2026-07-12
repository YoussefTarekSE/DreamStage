import { AudioBars } from "@/components/ui/AudioBars";

export function VinylLoader({ label = "Generating" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center gap-5 py-8 text-center" role="status" aria-live="polite">
      <div className="relative h-28 w-28">
        <div
          className="absolute inset-0 rounded-full border border-white/10 bg-[radial-gradient(circle_at_center,#0b0f0d_0_18%,#2a302d_19%_22%,#0d1211_23%_56%,#222b27_57%_61%,#0a0f0d_62%)] shadow-[0_18px_70px_rgba(0,0,0,0.45)]"
          style={{ animation: "vinyl 2.8s linear infinite" }}
        />
        <div className="absolute inset-[2.15rem] rounded-full border border-emerald-300/30 bg-emerald-300/20" />
        <div className="absolute right-0 top-3 h-16 w-2 origin-bottom rotate-[-24deg] rounded-full bg-white/40" />
      </div>
      <div className="space-y-2">
        <p className="text-lg font-semibold text-white">{label}</p>
        <AudioBars bars={14} />
      </div>
    </div>
  );
}
