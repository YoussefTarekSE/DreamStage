import { describe, expect, it } from "vitest";
import {
  normalizeEmail,
  validateEmail,
  validatePassword,
  validateUsername,
} from "./auth-validation";

describe("auth validation", () => {
  it("normalizes and validates email addresses", () => {
    expect(normalizeEmail("  ARTIST@Example.COM ")).toBe("artist@example.com");
    expect(validateEmail("artist@example.com").valid).toBe(true);
    expect(validateEmail("bad address").valid).toBe(false);
  });

  it("requires production-strength passwords", () => {
    expect(validatePassword("DreamStage1!").valid).toBe(true);
    expect(validatePassword("short1!A").valid).toBe(false);
    expect(validatePassword("dreamstage1!").requirements.uppercase).toBe(false);
    expect(validatePassword("DREAMSTAGE1!").requirements.lowercase).toBe(false);
    expect(validatePassword("DreamStage!!").requirements.number).toBe(false);
    expect(validatePassword("DreamStage11").requirements.special).toBe(false);
  });

  it("allows only safe usernames and rejects offensive words", () => {
    expect(validateUsername("vocal_hero").valid).toBe(true);
    expect(validateUsername("vo").valid).toBe(false);
    expect(validateUsername("name with spaces").valid).toBe(false);
    expect(validateUsername("badword_artist").valid).toBe(false);
  });
});
