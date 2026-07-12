import Link from "next/link";
import { Logo } from "@/components/ui/Logo";

export default function LandingPage() {
  return (
    <main className="flex min-h-screen flex-col bg-zinc-950">
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-5 border-b border-zinc-900">
        <Logo size="md" />
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="text-sm text-zinc-400 hover:text-white transition-colors px-4 py-2"
          >
            Log In
          </Link>
          <Link
            href="/signup"
            className="text-sm bg-violet-600 text-white hover:bg-violet-500 transition-colors px-4 py-2 rounded-lg font-semibold"
          >
            Get Started Free
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
        <div className="max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-xs text-violet-300">
            <span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />
            Beta — Free for early artists
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">
            Your voice deserves a{" "}
            <span className="text-violet-400">professional producer</span>
          </h1>
          <p className="text-lg text-zinc-400 max-w-xl mx-auto leading-relaxed">
            DreamStage gives every person with a voice and a dream the creative
            partner they never had — without ever replacing the art that makes
            them uniquely themselves.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-4">
            <Link
              href="/signup"
              className="w-full sm:w-auto text-base bg-violet-600 text-white hover:bg-violet-500 transition-colors px-8 py-4 rounded-lg font-semibold"
            >
              Start Making Music Free
            </Link>
            <Link
              href="/login"
              className="w-full sm:w-auto text-base text-zinc-400 hover:text-white transition-colors px-8 py-4 rounded-lg border border-zinc-800 hover:border-zinc-700"
            >
              I have an account
            </Link>
          </div>
        </div>
      </section>

      {/* Three Laws */}
      <section className="border-t border-zinc-900 px-6 py-16">
        <div className="mx-auto max-w-4xl grid grid-cols-1 gap-8 sm:grid-cols-3">
          {[
            {
              title: "Your Art Is Untouchable",
              body: "The AI never generates music on your behalf. It serves you — as your producer, engineer, and coach.",
            },
            {
              title: "You're Always In Control",
              body: "No black boxes. The AI explains every decision in plain language. Every change is yours to approve.",
            },
            {
              title: "Zero Knowledge Required",
              body: "You don't need to know anything about music production. If you have a voice, you're ready.",
            },
          ].map((card) => (
            <div
              key={card.title}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 space-y-3"
            >
              <div className="h-8 w-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
                <div className="h-3 w-3 rounded-full bg-violet-400" />
              </div>
              <h3 className="font-semibold text-white">{card.title}</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">{card.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-zinc-900 px-6 py-6 text-center text-xs text-zinc-600">
        © 2026 DreamStage. Built for artists who never got a chance.
      </footer>
    </main>
  );
}
