import { NextResponse, type NextRequest } from "next/server";

// Next.js 16 renamed Middleware → Proxy. Per the docs, Proxy must NOT do slow
// data fetching or full session management — only fast, optimistic checks.
//
// The previous middleware called supabase.auth.getUser() (a network round-trip
// to Supabase) on EVERY request. When Supabase was slow/unreachable, auth-js
// retried with exponential backoff for ~25s, freezing every page load. That
// network auth was also redundant: every protected page already calls
// getUser() and redirect("/login") itself, and the API backend validates the
// JWT on each request. So real authorization is enforced at the data layer.
//
// Here we only do an OPTIMISTIC redirect based on the presence of a Supabase
// session cookie — zero network, instant, and resilient to auth-server outages.

const PROTECTED_PREFIXES = ["/dashboard", "/studio", "/profile", "/onboarding"];
const AUTH_ROUTES = ["/login", "/signup"];

function hasSessionCookie(request: NextRequest): boolean {
  // @supabase/ssr stores the session in cookies named `sb-<ref>-auth-token`
  // (chunked as `…-auth-token.0`, `.1` for large tokens). Presence is enough
  // for an optimistic check; validity is verified later by the page + backend.
  return request.cookies
    .getAll()
    .some((c) => c.name.startsWith("sb-") && c.name.includes("-auth-token") && Boolean(c.value));
}

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isProtected = PROTECTED_PREFIXES.some((r) => pathname.startsWith(r));
  const isAuthRoute = AUTH_ROUTES.includes(pathname);

  // Anything that is neither protected nor an auth route never needs the proxy.
  if (!isProtected && !isAuthRoute) {
    return NextResponse.next();
  }

  const signedIn = hasSessionCookie(request);

  if (!signedIn && isProtected) {
    return NextResponse.redirect(new URL("/login", request.url));
  }
  if (signedIn && isAuthRoute) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return NextResponse.next();
}

// Run only on the routes that actually gate on auth — the public site (home,
// marketing, static assets) is never touched by the proxy.
export const config = {
  matcher: [
    "/dashboard/:path*",
    "/studio/:path*",
    "/profile/:path*",
    "/onboarding/:path*",
    "/login",
    "/signup",
  ],
};
