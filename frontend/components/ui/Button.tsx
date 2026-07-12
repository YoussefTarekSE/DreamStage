import { ButtonHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  loading?: boolean;
  fullWidth?: boolean;
}

export function Button({
  variant = "primary",
  loading = false,
  fullWidth = false,
  children,
  className = "",
  disabled,
  ...props
}: ButtonProps) {
  const base =
    "button-ripple inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-5 py-3 text-sm font-semibold transition-all duration-200 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50";

  const variants = {
    primary: "bg-emerald-400 text-emerald-950 shadow-[0_14px_36px_rgba(48,224,161,0.22)] hover:bg-emerald-300",
    secondary: "border border-white/10 bg-white/[0.07] text-white hover:border-white/18 hover:bg-white/[0.11]",
    ghost: "text-zinc-300 hover:bg-white/[0.07] hover:text-white",
    danger: "bg-rose-500 text-white shadow-[0_14px_36px_rgba(244,63,94,0.2)] hover:bg-rose-400",
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${fullWidth ? "w-full" : ""} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          <span>Working</span>
        </>
      ) : (
        children
      )}
    </button>
  );
}
