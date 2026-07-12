export function AudioBars({ active = true, bars = 18 }: { active?: boolean; bars?: number }) {
  return (
    <div className="flex h-12 items-end justify-center gap-1" aria-hidden="true">
      {Array.from({ length: bars }).map((_, index) => (
        <span
          key={index}
          className="block w-1 rounded-full bg-gradient-to-t from-emerald-400 via-sky-300 to-amber-200"
          style={{
            height: `${18 + ((index * 11) % 28)}px`,
            opacity: active ? 0.9 : 0.32,
            transformOrigin: "bottom",
            animation: active ? `bars ${760 + index * 35}ms ease-in-out infinite` : undefined,
            animationDelay: `${index * 58}ms`,
          }}
        />
      ))}
    </div>
  );
}
