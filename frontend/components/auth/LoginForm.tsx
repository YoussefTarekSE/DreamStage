"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Globe, Eye, EyeOff, Lock, Mail, Send } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { normalizeEmail, validateEmail } from "@/lib/auth-validation";
import { useLanguage } from "@/hooks/useLanguage";
import { AmbientStage } from "@/components/ui/AmbientStage";
import { AudioBars } from "@/components/ui/AudioBars";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Logo } from "@/components/ui/Logo";

export function LoginForm() {
  const router = useRouter();
  const { t, isRTL } = useLanguage();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setNotice("");

    const emailValidation = validateEmail(email);
    if (!emailValidation.valid) return setError(emailValidation.message ?? t.emailRequired);
    if (!password) return setError(t.passwordRequired);

    setLoading(true);
    const supabase = createClient();
    const { error: authError } = await supabase.auth.signInWithPassword({
      email: normalizeEmail(email),
      password,
    });

    if (authError) {
      setError(t.invalidCredentials);
      setLoading(false);
      return;
    }

    if (!remember) {
      window.sessionStorage.setItem("dreamstage-session-only", "true");
    }

    router.push("/dashboard");
    router.refresh();
  }

  async function handleGoogleLogin() {
    setGoogleLoading(true);
    const supabase = createClient();
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
  }

  async function handleForgotPassword() {
    setError("");
    setNotice("");
    const emailValidation = validateEmail(email);
    if (!emailValidation.valid) {
      setError("Enter your email first, then request the reset link.");
      return;
    }
    setResetLoading(true);
    const supabase = createClient();
    const { error: resetError } = await supabase.auth.resetPasswordForEmail(normalizeEmail(email), {
      redirectTo: `${window.location.origin}/login`,
    });
    setResetLoading(false);
    if (resetError) {
      setError("Could not send the reset email. Try again in a moment.");
      return;
    }
    setNotice("Password reset link sent. Check your inbox.");
  }

  return (
    <AmbientStage>
      <main className="grid min-h-[100dvh] items-center gap-8 px-4 py-10 md:grid-cols-[1fr_minmax(360px,460px)] md:px-8 lg:px-14" dir={isRTL ? "rtl" : "ltr"}>
        <section className="hidden max-w-2xl md:block">
          <Logo size="lg" />
          <div className="mt-10">
            <p className="text-sm font-semibold uppercase text-emerald-200">AI production workspace</p>
            <h1 className="mt-4 text-5xl font-semibold leading-tight text-white">
              Step into a studio that remembers your voice.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-zinc-300">
              Sign in to continue your songs, saved producer cuts, coaching notes, and final mixes.
            </p>
          </div>
          <div className="mt-10 max-w-sm rounded-lg border border-white/10 bg-white/[0.05] p-5">
            <AudioBars />
            <p className="mt-4 text-sm text-zinc-300">Session sync, secure auth, and persistent project history.</p>
          </div>
        </section>

        <section className="glass-panel-strong mx-auto w-full max-w-[460px] rounded-lg p-5 sm:p-7">
          <div className="mb-7 md:hidden">
            <Logo size="md" />
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-white">Welcome back</h2>
            <p className="mt-2 text-sm text-zinc-400">{t.tagline}</p>
          </div>

          <form onSubmit={handleLogin} className="mt-7 space-y-4">
            <Input
              label="Email"
              type="email"
              placeholder={t.emailPlaceholder}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              icon={<Mail className="h-4 w-4" aria-hidden="true" />}
            />
            <Input
              label="Password"
              type={showPassword ? "text" : "password"}
              placeholder={t.passwordPlaceholder}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              icon={<Lock className="h-4 w-4" aria-hidden="true" />}
              action={
                <button
                  type="button"
                  className="rounded-md p-1 text-zinc-400 transition-colors hover:text-white"
                  onClick={() => setShowPassword((value) => !value)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              }
            />

            <div className="flex items-center justify-between gap-3 text-sm">
              <label className="flex items-center gap-2 text-zinc-300">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                  className="h-4 w-4 rounded border-white/20 bg-white/10 accent-emerald-300"
                />
                Remember me
              </label>
              <button
                type="button"
                onClick={handleForgotPassword}
                className="inline-flex items-center gap-1 text-emerald-200 transition-colors hover:text-emerald-100"
                disabled={resetLoading}
              >
                <Send className="h-3.5 w-3.5" aria-hidden="true" />
                Forgot password
              </button>
            </div>

            {error && <p className="rounded-lg border border-rose-300/20 bg-rose-300/10 px-3 py-2 text-sm text-rose-100">{error}</p>}
            {notice && <p className="rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-3 py-2 text-sm text-emerald-100">{notice}</p>}

            <Button type="submit" fullWidth loading={loading}>
              {t.logIn}
            </Button>
          </form>

          <div className="my-6 flex items-center gap-3 text-xs text-zinc-500">
            <div className="h-px flex-1 bg-white/10" />
            or
            <div className="h-px flex-1 bg-white/10" />
          </div>

          <Button variant="secondary" fullWidth loading={googleLoading} onClick={handleGoogleLogin}>
            <Globe className="h-4 w-4" aria-hidden="true" />
            {t.continueWithGoogle}
          </Button>

          <p className="mt-6 text-center text-sm text-zinc-400">
            {t.noAccount}{" "}
            <Link href="/signup" className="font-medium text-emerald-200 hover:text-emerald-100">
              {t.createAccount}
            </Link>
          </p>
        </section>
      </main>
    </AmbientStage>
  );
}
