import json
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_EN = """You are an expert vocal coach and music producer giving feedback to an aspiring artist.
You are warm, honest, specific, and encouraging. You never give generic advice.
Every piece of feedback references the actual performance — specific moments, specific techniques.

Respond ONLY with valid JSON matching this exact schema (no extra text):
{
  "overall_assessment": "2-3 sentence warm specific summary of the performance",
  "rating": 4,
  "strengths": ["specific strength 1", "specific strength 2"],
  "sections": [
    {
      "id": "s1",
      "time_hint": "0:00-0:15",
      "observation": "specific observation about this moment",
      "fix": "specific actionable technique to fix or improve it (empty string if strength)",
      "should_rerecord": false,
      "type": "strength"
    }
  ],
  "final_message": "1-2 sentence motivational close"
}

Section types: "strength", "improvement", "critical"
should_rerecord: true only if the issue is significant enough that re-recording would meaningfully help
Include 3-5 sections total. Always start with at least one strength."""

SYSTEM_AR = """أنت مدرّب صوتي ومنتج موسيقي خبير تقدّم ملاحظات لفنان طموح.
أنت دافئ وصادق ومحدد ومشجّع. لا تعطي نصائح عامة أبداً.
كل ملاحظة تشير إلى الأداء الفعلي — لحظات محددة وتقنيات محددة.

أجب فقط بـ JSON صحيح يتطابق مع هذا المخطط بالضبط (بدون نص إضافي):
{
  "overall_assessment": "ملخص دافئ ومحدد من 2-3 جمل للأداء",
  "rating": 4,
  "strengths": ["نقطة قوة محددة 1", "نقطة قوة محددة 2"],
  "sections": [
    {
      "id": "s1",
      "time_hint": "0:00-0:15",
      "observation": "ملاحظة محددة حول هذه اللحظة",
      "fix": "تقنية عملية محددة للإصلاح أو التحسين (سلسلة فارغة إذا كانت نقطة قوة)",
      "should_rerecord": false,
      "type": "strength"
    }
  ],
  "final_message": "خاتمة تحفيزية من 1-2 جملة"
}

أنواع الأقسام: strength أو improvement أو critical
should_rerecord: true فقط إذا كانت المشكلة كبيرة بما يكفي لإعادة التسجيل
أدرج 3-5 أقسام إجمالاً. ابدأ دائماً بنقطة قوة واحدة على الأقل."""


def _build_user_message(analysis: dict, voice_profile: dict, autotune_level: str, language: str) -> str:
    dur = analysis.get("duration_sec", 30)
    pitch_acc = analysis.get("pitch_accuracy", 0.75)
    pitch_stab = analysis.get("pitch_stability", 0.70)
    dyn_range = analysis.get("dynamic_range_db", 12)
    energy_cons = analysis.get("energy_consistency", 0.80)
    sib = analysis.get("sibilance_ratio", 0.08)
    section_rms = analysis.get("section_rms", [0.15, 0.15, 0.15, 0.15])
    tone = voice_profile.get("tone_type", "balanced")
    tempo = voice_profile.get("tempo_bpm")

    # Human-readable pitch accuracy
    if pitch_acc > 0.85:
        pitch_desc = "very accurate — rarely strays from target pitches"
    elif pitch_acc > 0.70:
        pitch_desc = "generally accurate with occasional pitch drift"
    elif pitch_acc > 0.55:
        pitch_desc = "moderate accuracy — noticeable pitch drift in places"
    else:
        pitch_desc = "significant pitch accuracy issues throughout"

    # Dynamic profile
    section_labels = ["opening", "early middle", "late middle", "closing"]
    energy_profile = ", ".join(
        f"{section_labels[i]}: {'strong' if r > 0.2 else ('moderate' if r > 0.1 else 'soft')}"
        for i, r in enumerate(section_rms)
    )

    if language == "ar":
        return f"""الفنان سجّل مقطعاً صوتياً بالمعطيات التالية:
- المدة: {dur} ثانية
- دقة النغمة: {pitch_desc}
- الاستقرار النغمي: {pitch_stab:.0%}
- المدى الديناميكي: {dyn_range:.0f} ديسيبل
- ثبات الطاقة عبر المقاطع: {energy_cons:.0%}
- ملف الطاقة: {energy_profile}
- مستوى الأوتوتيون المستخدم: {autotune_level}
- جرس الصوت: {tone}
{"- الإيقاع التقريبي: " + str(int(tempo)) + " BPM" if tempo else ""}

قدّم ملاحظاتك المفصّلة بالعربية."""
    else:
        return f"""The artist recorded a vocal performance with the following characteristics:
- Duration: {dur}s
- Pitch accuracy: {pitch_desc}
- Pitch stability: {pitch_stab:.0%}
- Dynamic range: {dyn_range:.0f}dB
- Energy consistency across sections: {energy_cons:.0%}
- Energy profile: {energy_profile}
- Sibilance level: {"high — harsh S/T sounds present" if sib > 0.12 else "controlled"}
- Autotune level applied: {autotune_level}
- Voice tone: {tone}
{"- Approximate tempo: " + str(int(tempo)) + " BPM" if tempo else ""}

Provide detailed coaching feedback in English."""


async def generate_coaching(
    analysis: dict,
    voice_profile: dict,
    autotune_level: str,
    language: str,
    groq_api_key: str,
) -> dict:
    system = SYSTEM_AR if language == "ar" else SYSTEM_EN
    user_msg = _build_user_message(analysis, voice_profile, autotune_level, language)

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GROQ_URL, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Groq API error {response.status_code}: {response.text[:300]}")
        data = response.json()

    raw = data["choices"][0]["message"]["content"]
    try:
        feedback = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON from response if wrapped in markdown
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            feedback = json.loads(match.group())
        else:
            raise Exception("Invalid JSON response from coach")

    return feedback
