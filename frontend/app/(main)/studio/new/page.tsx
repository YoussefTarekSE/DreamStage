import { createClient, getUserSafe } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import { Logo } from "@/components/ui/Logo";
import Link from "next/link";
import { StudioFlow } from "@/components/studio/StudioFlow";

export default async function NewStudioPage() {
  const supabase = await createClient();
  const { user } = await getUserSafe(supabase);
  if (!user) redirect("/login");

  // Must have a voice profile first
  const { data: voiceProfile } = await supabase
    .from("voice_profiles")
    .select("id")
    .eq("user_id", user.id)
    .maybeSingle();

  if (!voiceProfile) redirect("/onboarding/voice-training");

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col">
      <header className="flex items-center justify-between border-b border-zinc-900 px-6 py-4">
        <Logo size="sm" />
        <Link href="/dashboard" className="text-sm text-zinc-500 hover:text-white transition-colors">
          ← {" "}My Songs
        </Link>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center px-6 py-12">
        <StudioFlow />
      </main>
    </div>
  );
}
