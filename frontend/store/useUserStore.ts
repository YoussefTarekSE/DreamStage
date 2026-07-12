import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Language = "en" | "ar";

interface UserStore {
  language: Language;
  setLanguage: (lang: Language) => void;
}

export const useUserStore = create<UserStore>()(
  persist(
    (set) => ({
      language: "en",
      setLanguage: (language) => set({ language }),
    }),
    { name: "dreamstage-user" }
  )
);
