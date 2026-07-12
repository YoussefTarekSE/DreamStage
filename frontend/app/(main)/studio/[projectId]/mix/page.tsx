import { createClient, getUserSafe } from "@/lib/supabase/server";
import { redirect, notFound } from "next/navigation";
import { Logo } from "@/components/ui/Logo";
import Link from "next/link";
import { MixFlow } from "@/components/studio/MixFlow";

export default async function MixPage({
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
    .select("id, name, status, final_mp3_key, final_wav_key")
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
              <span className={i === 3 ? "text-violet-400 font-medium" : "text-zinc-400"}>{s}</span>
            </div>
          ))}
        </div>
      </div>

      <main className="flex flex-1 flex-col items-center justify-center px-6 py-12">
        <MixFlow
          projectId={projectId}
          projectName={project.name}
          cachedMp3Url={project.status === "completed" && project.final_mp3_key ? undefined : undefined}
          hasCachedMix={project.status === "completed" && Boolean(project.final_mp3_key)}
        />
      </main>
    </div>
  );
}
