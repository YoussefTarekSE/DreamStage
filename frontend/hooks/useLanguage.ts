import { useUserStore } from "@/store/useUserStore";

const translations = {
  en: {
    appName: "DreamStage",
    tagline: "Your voice. Your art. Professional.",
    signUp: "Get Started Free",
    logIn: "Log In",
    emailPlaceholder: "Email address",
    passwordPlaceholder: "Password",
    continueWithGoogle: "Continue with Google",
    alreadyHaveAccount: "Already have a DreamStage account?",
    noAccount: "New to DreamStage?",
    createAccount: "Create your account",
    chooseLanguage: "Choose your language",
    english: "English",
    arabic: "العربية",
    dashboard: "My Songs",
    noProjects: "No songs yet. Let's make your first one.",
    startRecording: "Start Recording",
    loading: "Loading...",
    errorGeneric: "Something went wrong. Please try again.",
    invalidCredentials: "Invalid email or password.",
    emailRequired: "Email is required.",
    passwordRequired: "Password is required.",
    passwordTooShort: "Password must be at least 6 characters.",
  },
  ar: {
    appName: "DreamStage",
    tagline: "صوتك. فنّك. احترافي.",
    signUp: "ابدأ مجاناً",
    logIn: "تسجيل الدخول",
    emailPlaceholder: "البريد الإلكتروني",
    passwordPlaceholder: "كلمة المرور",
    continueWithGoogle: "المتابعة مع Google",
    alreadyHaveAccount: "لديك حساب على DreamStage؟",
    noAccount: "جديد على DreamStage؟",
    createAccount: "إنشاء حسابك",
    chooseLanguage: "اختر لغتك",
    english: "English",
    arabic: "العربية",
    dashboard: "أغاني",
    noProjects: "لا توجد أغاني بعد. لنصنع أولى أغانيك.",
    startRecording: "ابدأ التسجيل",
    loading: "جار التحميل...",
    errorGeneric: "حدث خطأ ما. يرجى المحاولة مرة أخرى.",
    invalidCredentials: "البريد الإلكتروني أو كلمة المرور غير صحيحة.",
    emailRequired: "البريد الإلكتروني مطلوب.",
    passwordRequired: "كلمة المرور مطلوبة.",
    passwordTooShort: "يجب أن تتكون كلمة المرور من 6 أحرف على الأقل.",
  },
} as const;

export function useLanguage() {
  const { language, setLanguage } = useUserStore();
  const t = translations[language];
  const isRTL = language === "ar";
  return { t, language, setLanguage, isRTL };
}
