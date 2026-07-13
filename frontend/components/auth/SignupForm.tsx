"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { BadgeCheck, Globe, Eye, EyeOff, Lock, Mail, UserRound } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import {
  normalizeEmail,
  normalizeUsername,
  validateEmail,
  validatePassword,
  validateUsername,
} from "@/lib/auth-validation";
import { useLanguage } from "@/hooks/useLanguage";
import { useUserStore } from "@/store/useUserStore";
import { AmbientStage } from "@/components/ui/AmbientStage";
import { AudioBars } from "@/components/ui/AudioBars";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Logo } from "@/components/ui/Logo";

type Step = "form" | "language" | "verify";
type Availability = "idle" | "checking" | "available" | "taken";

export function SignupForm() {
  const router = useRouter();
  const { t, isRTL } = useLanguage();
  const { setLanguage } = useUserStore();
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [step, setStep] = useState<Step>(() =>
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("step") === "language"
      ? "language"
      : "form"
  );
  const [availability, setAvailability] = useState<Availability>("idle");
  const [checkedFor, setCheckedFor] = useState("");

  const passwordValidation = useMemo(() => validatePassword(password), [password]);
  const usernameValidation = useMemo(() => validateUsername(username), [username]);
  const normalizedUsername = normalizeUsername(username);

  // Adjust availability during render when the relevant inputs change, instead
  // of resetting it from inside the effect below (avoids an extra render pass).
  if (checkedFor !== normalizedUsername) {
    setCheckedFor(normalizedUsername);
    setAvailability(username && usernameValidation.valid ? "checking" : "idle");
  }

  useEffect(() => {
    if (!username || !usernameValidation.valid) return;
    let cancelled = false;

    const timer = window.setTimeout(async () => {
      const supabase = createClient();
      const { data, error: rpcError } = await supabase.rpc("is_username_available", {
        candidate: normalizedUsername,
      });
      if (cancelled) return;
      if (rpcError) {
        setAvailability("idle");
        return;
      }
      setAvailability(data ? "available" : "taken");
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [normalizedUsername, username, usernameValidation.valid]);

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    setError("");

    const emailValidation = validateEmail(email);
    if (!emailValidation.valid) return setError(emailValidation.message ?? t.emailRequired);
    if (!usernameValidation.valid) return setError(usernameValidation.message ?? "Choose a valid username.");
    if (availability === "taken") return setError("That username is already taken.");
    if (!displayName.trim()) return setError("Display name is required.");
    if (!passwordValidation.valid) return setError(passwordValidation.message ?? t.passwordTooShort);

    setLoading(true);
    const supabase = createClient();
    const { data, error: authError } = await supabase.auth.signUp({
      email: normalizeEmail(email),
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback?onboarding=true`,
        data: {
          username: normalizedUsername,
          display_name: displayName.trim(),
        },
      },
    });

    if (authError) {
      setError(authError.message);
      setLoading(false);
      return;
    }

    setLoading(false);
    if (data.session) {
      setStep("language");
      return;
    }
    setStep("verify");
  }

  async function handleGoogleSignup() {
    setGoogleLoading(true);
    const supabase = createClient();
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback?onboarding=true` },
    });
  }

  async function handleLanguageSelect(lang: "en" | "ar") {
    setLanguage(lang);
    const supabase = createClient();
    const { data } = await supabase.auth.getUser();
    if (data.user) {
      await supabase.from("user_settings").upsert({ user_id: data.user.id, language: lang });
    }
    router.push("/dashboard");
    router.refresh();
  }

  if (step === "verify") {
    return (
      <AmbientStage density="calm">
        <main className="flex min-h-[100dvh] items-center justify-center px-4">
          <section className="glass-panel-strong w-full max-w-md rounded-lg p-7 text-center">
            <div className="mx-auto grid h-14 w-14 place-items-center rounded-lg border border-emerald-300/20 bg-emerald-300/10">
              <BadgeCheck className="h-7 w-7 text-emerald-200" aria-hidden="true" />
            </div>
            <h1 className="mt-5 text-2xl font-semibold text-white">Verify your email</h1>
            <p className="mt-3 text-sm leading-6 text-zinc-400">
              We sent a verification link to {normalizeEmail(email)}. Open it to activate your DreamStage account.
            </p>
            <Link href="/login" className="mt-6 inline-flex min-h-11 items-center justify-center rounded-lg bg-emerald-400 px-5 py-3 text-sm font-semibold text-emerald-950 hover:bg-emerald-300">
              Back to login
            </Link>
          </section>
        </main>
      </AmbientStage>
    );
  }

  if (step === "language") {
    return (
      <AmbientStage density="calm">
        <main className="flex min-h-[100dvh] items-center justify-center px-4">
          <section className="glass-panel-strong w-full max-w-md rounded-lg p-7 text-center">
            <Logo size="md" />
            <h1 className="mt-6 text-2xl font-semibold text-white">{t.chooseLanguage}</h1>
            <p className="mt-2 text-sm text-zinc-400">Your language preference is saved to your account settings.</p>
            <div className="mt-7 grid gap-3">
              <Button variant="secondary" fullWidth onClick={() => handleLanguageSelect("en")}>
                {t.english}
              </Button>
              <Button variant="secondary" fullWidth onClick={() => handleLanguageSelect("ar")}>
                {t.arabic}
              </Button>
            </div>
          </section>
        </main>
      </AmbientStage>
    );
  }

  const usernameError =
    username && !usernameValidation.valid
      ? usernameValidation.message
      : availability === "taken"
        ? "That username is already taken."
        : undefined;
  const usernameHint =
    availability === "checking"
      ? "Checking availability..."
      : availability === "available"
        ? "Username available."
        : "3-20 letters, numbers, and underscores.";

  return (
    <AmbientStage>
      <main className="grid min-h-[100dvh] items-center gap-8 px-4 py-10 md:grid-cols-[1fr_minmax(390px,520px)] md:px-8 lg:px-14" dir={isRTL ? "rtl" : "ltr"}>
        <section className="hidden max-w-2xl md:block">
          <Logo size="lg" />
          <div className="mt-10">
            <p className="text-sm font-semibold uppercase text-emerald-200">Create without losing yourself</p>
            <h1 className="mt-4 text-5xl font-semibold leading-tight text-white">
              Build a profile your producer can actually learn from.
            </h1>
            <p className="mt-5 max-w-xl text-base leading-7 text-zinc-300">
              Your voice profile, preferences, credits, history, favorites, and settings persist across every session.
            </p>
          </div>
          <div className="mt-10 max-w-sm rounded-lg border border-white/10 bg-white/[0.05] p-5">
            <AudioBars />
            <p className="mt-4 text-sm text-zinc-300">Strong password rules and username validation are active before signup.</p>
          </div>
        </section>

        <section className="glass-panel-strong mx-auto w-full max-w-[520px] rounded-lg p-5 sm:p-7">
          <div className="mb-7 md:hidden">
            <Logo size="md" />
          </div>
          <h2 className="text-2xl font-semibold text-white">Create your account</h2>
          <p className="mt-2 text-sm text-zinc-400">{t.tagline}</p>

          <form onSubmit={handleSignup} className="mt-7 space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Input
                label="Display name"
                placeholder="Maya Stone"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                autoComplete="name"
                icon={<UserRound className="h-4 w-4" aria-hidden="true" />}
              />
              <Input
                label="Username"
                placeholder="maya_voice"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                icon={<BadgeCheck className="h-4 w-4" aria-hidden="true" />}
                error={usernameError}
                hint={usernameHint}
              />
            </div>

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
              placeholder="10+ characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
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

            <div className="space-y-3 rounded-lg border border-white/10 bg-white/[0.05] p-3">
              <div className="h-2 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-rose-300 via-amber-200 to-emerald-300 transition-all"
                  style={{ width: `${(passwordValidation.score / 5) * 100}%` }}
                />
              </div>
              <div className="grid gap-2 text-xs text-zinc-400 sm:grid-cols-2">
                {[
                  ["10+ characters", passwordValidation.requirements.length],
                  ["Uppercase letter", passwordValidation.requirements.uppercase],
                  ["Lowercase letter", passwordValidation.requirements.lowercase],
                  ["Number", passwordValidation.requirements.number],
                  ["Special character", passwordValidation.requirements.special],
                ].map(([label, passed]) => (
                  <span key={String(label)} className={passed ? "text-emerald-200" : "text-zinc-500"}>
                    {passed ? "OK" : "--"} {label}
                  </span>
                ))}
              </div>
            </div>

            {error && <p className="rounded-lg border border-rose-300/20 bg-rose-300/10 px-3 py-2 text-sm text-rose-100">{error}</p>}

            <Button type="submit" fullWidth loading={loading}>
              {t.signUp}
            </Button>
          </form>

          <div className="my-6 flex items-center gap-3 text-xs text-zinc-500">
            <div className="h-px flex-1 bg-white/10" />
            or
            <div className="h-px flex-1 bg-white/10" />
          </div>

          <Button variant="secondary" fullWidth loading={googleLoading} onClick={handleGoogleSignup}>
            <Globe className="h-4 w-4" aria-hidden="true" />
            {t.continueWithGoogle}
          </Button>

          <p className="mt-6 text-center text-sm text-zinc-400">
            {t.alreadyHaveAccount}{" "}
            <Link href="/login" className="font-medium text-emerald-200 hover:text-emerald-100">
              {t.logIn}
            </Link>
          </p>
        </section>
      </main>
    </AmbientStage>
  );
}
