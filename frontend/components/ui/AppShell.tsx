import Link from "next/link";
import { Home, Music2, Settings, Sparkles, UserRound } from "lucide-react";
import { AmbientStage } from "@/components/ui/AmbientStage";
import { Logo } from "@/components/ui/Logo";

const navItems = [
  { href: "/dashboard", label: "Home", icon: Home },
  { href: "/studio/new", label: "Create", icon: Sparkles },
  { href: "/profile", label: "Profile", icon: UserRound },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function AppShell({
  children,
  title,
  subtitle,
  action,
  projectName,
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  projectName?: string;
}) {
  return (
    <AmbientStage density="calm">
      <div className="min-h-[100dvh] lg:grid lg:grid-cols-[260px_1fr]">
        <aside className="hidden border-r border-white/10 bg-black/18 px-5 py-5 backdrop-blur-xl lg:block">
          <Logo size="sm" />
          <nav className="mt-8 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-300 transition-colors hover:bg-white/[0.07] hover:text-white"
                >
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="mt-8 rounded-lg border border-white/10 bg-white/[0.05] p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <Music2 className="h-4 w-4 text-emerald-300" aria-hidden="true" />
              Live Studio
            </div>
            <p className="mt-2 text-xs leading-5 text-zinc-400">
              Voice-first production with persistent cuts, coaching, and mixes.
            </p>
          </div>
        </aside>

        <div className="flex min-h-[100dvh] flex-col">
          <header className="sticky top-0 z-20 border-b border-white/10 bg-[#060807]/80 px-4 py-4 backdrop-blur-xl sm:px-6">
            <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
              <div className="lg:hidden">
                <Logo size="sm" />
              </div>
              <div className="hidden min-w-0 lg:block">
                {title && <h1 className="truncate text-2xl font-semibold text-white">{title}</h1>}
                {subtitle && <p className="mt-1 max-w-2xl text-sm text-zinc-400">{subtitle}</p>}
                {projectName && <p className="text-sm text-zinc-400">{projectName}</p>}
              </div>
              <div className="ml-auto flex items-center gap-3">{action}</div>
            </div>
          </header>
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 pb-24 sm:px-6 lg:py-8">
            <div className="lg:hidden">
              {title && <h1 className="text-2xl font-semibold text-white">{title}</h1>}
              {subtitle && <p className="mt-2 text-sm text-zinc-400">{subtitle}</p>}
            </div>
            <div className={title ? "mt-6 lg:mt-0" : ""}>{children}</div>
          </main>
          <nav className="fixed inset-x-0 bottom-0 z-30 border-t border-white/10 bg-[#060807]/90 px-3 py-2 backdrop-blur-xl lg:hidden">
            <div className="mx-auto grid max-w-md grid-cols-4 gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex flex-col items-center gap-1 rounded-lg px-2 py-2 text-[11px] text-zinc-400 transition-colors hover:bg-white/[0.07] hover:text-white"
                  >
                    <Icon className="h-4 w-4" aria-hidden="true" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </nav>
        </div>
      </div>
    </AmbientStage>
  );
}
