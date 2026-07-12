import { ReactNode } from "react";

export function AmbientStage({ children, density = "full" }: { children: ReactNode; density?: "full" | "calm" }) {
  return (
    <div className="relative min-h-[100dvh] overflow-hidden bg-[#060807] text-white">
      <div className="pointer-events-none absolute inset-0 ambient-grid opacity-45" />
      <div
        className="pointer-events-none absolute -left-28 top-[-18rem] h-[34rem] w-[34rem] rounded-full bg-emerald-400/18 blur-3xl"
        style={{ animation: "float-slow 9s ease-in-out infinite" }}
      />
      <div
        className="pointer-events-none absolute right-[-10rem] top-28 h-[32rem] w-[32rem] rounded-full bg-sky-400/14 blur-3xl"
        style={{ animation: "float-slow 11s ease-in-out infinite reverse" }}
      />
      {density === "full" && (
        <div
          className="pointer-events-none absolute bottom-[-14rem] left-1/3 h-[30rem] w-[30rem] rounded-full bg-amber-300/12 blur-3xl"
          style={{ animation: "drift 14s ease-in-out infinite" }}
        />
      )}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
