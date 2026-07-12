import Link from "next/link";
import { Music2 } from "lucide-react";

export function EmptyState({
  title,
  body,
  href,
  action,
}: {
  title: string;
  body: string;
  href: string;
  action: string;
}) {
  return (
    <div className="glass-panel flex min-h-[360px] flex-col items-center justify-center rounded-lg px-6 py-16 text-center">
      <div className="grid h-16 w-16 place-items-center rounded-lg border border-emerald-300/20 bg-emerald-300/10">
        <Music2 className="h-7 w-7 text-emerald-200" aria-hidden="true" />
      </div>
      <h2 className="mt-5 text-xl font-semibold text-white">{title}</h2>
      <p className="mt-2 max-w-md text-sm leading-6 text-zinc-400">{body}</p>
      <Link
        href={href}
        className="button-ripple mt-6 inline-flex min-h-11 items-center justify-center rounded-lg bg-emerald-400 px-5 py-3 text-sm font-semibold text-emerald-950 shadow-[0_14px_36px_rgba(48,224,161,0.22)] transition-all hover:bg-emerald-300 active:scale-[0.98]"
      >
        {action}
      </Link>
    </div>
  );
}
