import { createClient, getUserSafe } from "@/lib/supabase/server";
import { redirect, notFound } from "next/navigation";
import { Logo } from "@/components/ui/Logo";
import Link from "next/link";
import { CoachFlow } from "@/components/studio/CoachFlow";

export default async function CoachPage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const supabase = await createClient();
  const { user } = await getUserSafe(supabase);
  if (!user) redirect("/login");

  const { data: project } = await supabase
    .from("projects")
    .select("id, name, coach_feedback, status")
    .eq("id", projectId)
    .eq("user_id", user.id)
    .maybeSingle();

  if (!project) notFound();

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col">
      <header className="flex items-center justify-between border-b border-zinc-900 px-6 py-4">
        <Logo size="sm" />
        <div className="flex items-center gap-4">
          <span className="text-sm text-zinc-400 truncate max-w-[200px]">{project.name}</span>
          <Link href="/dashboard" className="text-sm text-zinc-500 hover:text-white transition-colors">
            My Songs
          </Link>
        </div>
      </header>

      {/* Pipeline progress */}
      <div className="border-b border-zinc-900 px-6 py-3">
        <div className="mx-auto max-w-lg flex items-center gap-2 text-xs text-zinc-500">
          {["Vocal", "Beat", "Coach", "Mix"].map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              {i > 0 && <div className="h-px w-6 bg-zinc-800" />}
              <span className={i === 2 ? "text-violet-400 font-medium" : i < 2 ? "text-zinc-400" : ""}>{s}</span>
            </div>
          ))}
        </div>
      </div>

      <main className="flex flex-1 flex-col items-center justify-center px-6 py-12">
        <CoachFlow
          projectId={projectId}
          cachedFeedback={project.coach_feedback ?? undefined}
        />
      </main>
    </div>
  );
}
