"use client";

import { AlertTriangle, RotateCcw } from "lucide-react";
import { AmbientStage } from "@/components/ui/AmbientStage";
import { Button } from "@/components/ui/Button";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <AmbientStage density="calm">
      <main className="flex min-h-[100dvh] items-center justify-center px-6">
        <section className="glass-panel max-w-md rounded-lg p-8 text-center">
          <div className="mx-auto grid h-14 w-14 place-items-center rounded-lg border border-rose-300/20 bg-rose-300/10">
            <AlertTriangle className="h-7 w-7 text-rose-200" aria-hidden="true" />
          </div>
          <h1 className="mt-5 text-2xl font-semibold text-white">The session dropped</h1>
          <p className="mt-3 text-sm leading-6 text-zinc-400">
            DreamStage hit a recoverable problem. Your saved work is still protected.
          </p>
          <Button className="mt-6" onClick={reset}>
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Try again
          </Button>
        </section>
      </main>
    </AmbientStage>
  );
}
