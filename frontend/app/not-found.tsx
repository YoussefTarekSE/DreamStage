import Link from "next/link";
import { Compass } from "lucide-react";
import { AmbientStage } from "@/components/ui/AmbientStage";

export default function NotFound() {
  return (
    <AmbientStage density="calm">
      <main className="flex min-h-[100dvh] items-center justify-center px-6">
        <section className="glass-panel max-w-md rounded-lg p-8 text-center">
          <div className="mx-auto grid h-14 w-14 place-items-center rounded-lg border border-sky-300/20 bg-sky-300/10">
            <Compass className="h-7 w-7 text-sky-200" aria-hidden="true" />
          </div>
          <h1 className="mt-5 text-2xl font-semibold text-white">This stage is empty</h1>
          <p className="mt-3 text-sm leading-6 text-zinc-400">
            The page you opened does not exist or is no longer available.
          </p>
          <Link
            href="/dashboard"
            className="button-ripple mt-6 inline-flex min-h-11 items-center justify-center rounded-lg bg-emerald-400 px-5 py-3 text-sm font-semibold text-emerald-950 transition-all hover:bg-emerald-300 active:scale-[0.98]"
          >
            Back to dashboard
          </Link>
        </section>
      </main>
    </AmbientStage>
  );
}
