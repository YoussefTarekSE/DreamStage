export function Logo({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizes = { sm: "text-xl", md: "text-2xl", lg: "text-4xl" };
  return (
    <span className={`inline-flex items-center gap-2 font-bold text-white ${sizes[size]}`}>
      <span className="grid h-8 w-8 place-items-center rounded-lg border border-emerald-300/20 bg-emerald-300/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.12)]">
        <span className="h-3 w-3 rounded-full bg-emerald-300 shadow-[0_0_22px_rgba(48,224,161,0.65)]" />
      </span>
      <span>Dream<span className="text-emerald-300">Stage</span></span>
    </span>
  );
}
