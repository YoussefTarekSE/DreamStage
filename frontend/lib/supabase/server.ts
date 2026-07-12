import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

// Cap every Supabase network request so an unreachable/paused project (DNS
// ENOTFOUND, timeouts, blips) can never hang a server render. Without this a
// down auth host freezes the page for ~25s while auth-js retries.
const REQUEST_TIMEOUT_MS = 8000;

function timeoutFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  // Respect a caller-provided signal (auth-js sometimes passes its own);
  // otherwise attach a hard per-request timeout.
  if (init?.signal) return fetch(input, init);
  return fetch(input, { ...init, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
}

export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {}
        },
      },
      global: { fetch: timeoutFetch },
    }
  );
}

/**
 * Read the authenticated user with a hard deadline.
 *
 * supabase.auth.getUser() refreshes the token when it is stale, and if the auth
 * host is unreachable, auth-js retries that refresh with exponential backoff for
 * ~25s — freezing the server render. This bounds the whole call: on timeout (or
 * any error) it resolves to "signed out", so a protected page degrades to a fast
 * redirect to /login instead of hanging. When Supabase is healthy this returns
 * the real user well within the deadline.
 */
export async function getUserSafe(
  supabase: Awaited<ReturnType<typeof createClient>>,
  timeoutMs = 3500
) {
  try {
    return await Promise.race([
      supabase.auth.getUser().then((r) => ({ user: r.data.user })),
      new Promise<{ user: null }>((resolve) =>
        setTimeout(() => resolve({ user: null }), timeoutMs)
      ),
    ]);
  } catch {
    return { user: null };
  }
}
