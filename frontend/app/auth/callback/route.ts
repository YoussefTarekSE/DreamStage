import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  const onboarding = searchParams.get("onboarding");

  if (code) {
    const supabase = await createClient();
    await supabase.auth.exchangeCodeForSession(code);
  }

  const destination = onboarding ? "/signup?step=language" : "/dashboard";
  return NextResponse.redirect(`${origin}${destination}`);
}
