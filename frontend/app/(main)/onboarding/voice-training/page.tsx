import { createClient, getUserSafe } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import { VoiceTrainingFlow } from "@/components/voice-training/VoiceTrainingFlow";
import { Logo } from "@/components/ui/Logo";

export default async function VoiceTrainingPage() {
  const supabase = await createClient();
  const { user } = await getUserSafe(supabase);
  if (!user) redirect("/login");

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col">
      <header className="flex items-center px-6 py-4 border-b border-zinc-900">
        <Logo size="sm" />
      </header>
      <main className="flex flex-1 flex-col items-center justify-center px-6 py-12">
        <VoiceTrainingFlow />
      </main>
    </div>
  );
}
