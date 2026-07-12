export interface ValidationResult {
  valid: boolean;
  message?: string;
}

export interface PasswordValidation extends ValidationResult {
  requirements: {
    length: boolean;
    uppercase: boolean;
    lowercase: boolean;
    number: boolean;
    special: boolean;
  };
  score: number;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const USERNAME_RE = /^[A-Za-z0-9_]+$/;

const OFFENSIVE_WORDS = [
  "badword",
  "admin",
  "root",
  "support",
  "moderator",
  "staff",
];

export function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

export function validateEmail(value: string): ValidationResult {
  const email = normalizeEmail(value);
  if (!email) return { valid: false, message: "Email is required." };
  if (!EMAIL_RE.test(email)) return { valid: false, message: "Enter a valid email address." };
  return { valid: true };
}

export function validatePassword(value: string): PasswordValidation {
  const requirements = {
    length: value.length >= 10,
    uppercase: /[A-Z]/.test(value),
    lowercase: /[a-z]/.test(value),
    number: /[0-9]/.test(value),
    special: /[^A-Za-z0-9]/.test(value),
  };
  const score = Object.values(requirements).filter(Boolean).length;
  return {
    valid: score === Object.keys(requirements).length,
    message: score === Object.keys(requirements).length
      ? undefined
      : "Use 10+ characters with uppercase, lowercase, a number, and a symbol.",
    requirements,
    score,
  };
}

export function normalizeUsername(value: string): string {
  return value.trim().toLowerCase();
}

export function validateUsername(value: string): ValidationResult {
  const username = normalizeUsername(value);
  if (!username) return { valid: false, message: "Username is required." };
  if (username.length < 3) return { valid: false, message: "Username must be at least 3 characters." };
  if (username.length > 20) return { valid: false, message: "Username must be 20 characters or less." };
  if (!USERNAME_RE.test(username)) {
    return { valid: false, message: "Use letters, numbers, and underscores only." };
  }
  if (OFFENSIVE_WORDS.some((word) => username.includes(word))) {
    return { valid: false, message: "Choose a different username." };
  }
  return { valid: true };
}
