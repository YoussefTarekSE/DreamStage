import { createClient, getUserSafe } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import { Logo } from "@/components/ui/Logo";
import Link from "next/link";

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  created:          { label: "Created",       color: "text-zinc-400 border-zinc-700 bg-zinc-800/50" },
  voice_training:   { label: "Voice",         color: "text-blue-300 border-blue-500/30 bg-blue-500/10" },
  recording:        { label: "Vocal Ready",   color: "text-cyan-300 border-cyan-500/30 bg-cyan-500/10" },
  processing_vocal: { label: "Processing",    color: "text-yellow-300 border-yellow-500/30 bg-yellow-500/10" },
  beat_generation:  { label: "Beat Ready",    color: "text-orange-300 border-orange-500/30 bg-orange-500/10" },
  coaching:         { label: "Coach Done",    color: "text-fuchsia-300 border-fuchsia-500/30 bg-fuchsia-500/10" },
  mixing:           { label: "Mixing",        color: "text-violet-300 border-violet-500/30 bg-violet-500/10" },
  completed:        { label: "✓ Done",        color: "text-emerald-300 border-emerald-500/30 bg-emerald-500/10" },
};

const NEXT_STEP: Record<string, string> = {
  recording:       "/beat",
  beat_generation: "/beat",
  coaching:        "/coach",
  mixing:          "/mix",
};

export default async function DashboardPage() {
  const supabase = await createClient();
  const { user } = await getUserSafe(supabase);
  if (!user) redirect("/login");

  const { data: voiceProfile } = await supabase
    .from("voice_profiles")
    .select("id, tone_type")
    .eq("user_id", user.id)
    .maybeSingle();
  if (!voiceProfile) redirect("/onboarding/voice-training");

  const { data: projects } = await supabase
    .from("projects")
    .select("id, name, status, created_at, final_mp3_key")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  return (
    <div className="min-h-screen bg-zinc-950">
      <header className="flex items-center justify-between border-b border-zinc-900 px-6 py-4">
        <Logo size="sm" />
        <div className="flex items-center gap-4">
          {voiceProfile.tone_type && (
            <span className="hidden sm:block text-xs text-zinc-600 border border-zinc-800 rounded-full px-3 py-1">
              Voice: {voiceProfile.tone_type}
            </span>
          )}
          <form action="/auth/signout" method="post">
            <button type="submit" className="text-sm text-zinc-500 hover:text-white transition-colors">
              Sign out
            </button>
          </form>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-12">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold text-white">My Songs</h1>
          <Link
            href="/studio/new"
            className="inline-flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-500 transition-colors"
          >
            <span className="text-lg leading-none">+</span>
            New Song
          </Link>
        </div>

        {!projects || projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-zinc-800 py-24 text-center">
            <div className="mb-4 h-16 w-16 rounded-full bg-zinc-900 flex items-center justify-center text-3xl">🎤</div>
            <h2 className="text-lg font-semibold text-white mb-2">No songs yet</h2>
            <p className="text-sm text-zinc-500 mb-6 max-w-xs">
              Let&apos;s make your first one. Your voice, your lyrics, your art — professionally produced.
            </p>
            <Link href="/studio/new" className="inline-flex items-center gap-2 rounded-lg bg-violet-600 px-5 py-3 text-sm font-semibold text-white hover:bg-violet-500 transition-colors">
              Start Recording
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {projects.map((project) => {
              const statusInfo = STATUS_LABELS[project.status] ?? { label: project.status, color: "text-zinc-400 border-zinc-700" };
              const isCompleted = project.status === "completed";
              const nextStep = NEXT_STEP[project.status];

              return (
                <div key={project.id} className={`flex items-center justify-between rounded-xl border px-5 py-4 transition-all ${isCompleted ? "border-emerald-500/20 bg-emerald-500/5" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"}`}>
                  <div className="flex items-center gap-4 min-w-0">
                    <div className={`h-10 w-10 rounded-lg flex items-center justify-center text-lg shrink-0 ${isCompleted ? "bg-emerald-500/20" : "bg-violet-500/20"}`}>
                      {isCompleted ? "🎵" : "🎤"}
                    </div>
                    <div className="min-w-0">
                      <p className="font-medium text-white truncate">{project.name}</p>
                      <p className="text-xs text-zinc-500">
                        {new Date(project.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0 ml-3">
                    <span className={`text-xs px-2.5 py-1 rounded-full border ${statusInfo.color}`}>
                      {statusInfo.label}
                    </span>

                    {isCompleted && (
                      <Link
                        href={`/studio/${project.id}/mix`}
                        className="text-xs px-3 py-1.5 rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 transition-colors font-medium"
                      >
                        Download
                      </Link>
                    )}

                    {!isCompleted && nextStep && (
                      <Link
                        href={`/studio/${project.id}${nextStep}`}
                        className="text-xs px-3 py-1.5 rounded-lg bg-violet-600 text-white hover:bg-violet-500 transition-colors font-medium"
                      >
                        Continue →
                      </Link>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
