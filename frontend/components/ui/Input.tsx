import { InputHTMLAttributes, ReactNode } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  icon?: ReactNode;
  action?: ReactNode;
}

export function Input({ label, error, hint, icon, action, className = "", ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-sm font-medium text-zinc-200">{label}</label>
      )}
      <div className={`flex items-center gap-2 rounded-lg border bg-white/[0.06] px-3 transition-all focus-within:border-emerald-300/70 focus-within:bg-white/[0.09] ${error ? "border-rose-400/70" : "border-white/10"}`}>
        {icon && <span className="text-zinc-400">{icon}</span>}
        <input
          className={`min-h-11 w-full bg-transparent py-3 text-sm text-white placeholder:text-zinc-500 focus:outline-none disabled:opacity-50 ${className}`}
          {...props}
        />
        {action}
      </div>
      {hint && !error && <p className="text-xs text-zinc-500">{hint}</p>}
      {error && <p className="text-xs text-rose-300">{error}</p>}
    </div>
  );
}
