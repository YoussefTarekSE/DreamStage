import { createBrowserClient } from "@supabase/ssr";

// Cap every browser-side Supabase request so an unreachable/paused project can't
// freeze user actions (e.g. the login button) for ~25s while auth-js retries.
const REQUEST_TIMEOUT_MS = 8000;

function timeoutFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  if (init?.signal) return fetch(input, init);
  return fetch(input, { ...init, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
}

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { global: { fetch: timeoutFetch } }
  );
}
