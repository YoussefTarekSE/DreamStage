import { createClient } from "@/lib/supabase/client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getAuthHeaders(): Promise<Record<string, string>> {
  const supabase = createClient();
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("Not authenticated");
  return { Authorization: `Bearer ${token}` };
}

/** Throw a structured error for a failed response. Handles non-JSON bodies
 * (Render 502 HTML, proxy timeouts) instead of crashing the JSON parser. */
async function throwHttpError(res: Response): Promise<never> {
  let err: unknown;
  try {
    err = await res.json();
  } catch {
    err = {
      detail: {
        message_en: `The server had a problem (${res.status}). Please try again in a moment.`,
        message_ar: `حدثت مشكلة في الخادم (${res.status}). يرجى المحاولة بعد قليل.`,
      },
    };
  }
  throw err;
}

export async function checkRecordingQuality(blob: Blob): Promise<{
  ok: boolean;
  reason?: string;
  message_en?: string;
  message_ar?: string;
}> {
  const headers = await getAuthHeaders();
  const form = new FormData();
  form.append("file", blob, "recording.webm");
  const res = await fetch(`${API_URL}/api/voice-training/check-quality`, {
    method: "POST",
    headers,
    body: form,
  });
  return res.json();
}

export async function submitVoiceTraining(
  recordings: Blob[],
  language: string
): Promise<{
  status: string;
  voice_profile: Record<string, unknown>;
  message_en: string;
  message_ar: string;
}> {
  const headers = await getAuthHeaders();
  const form = new FormData();
  recordings.forEach((blob) => form.append("files", blob, "recording.webm"));
  form.append("language", language);
  const res = await fetch(`${API_URL}/api/voice-training/`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw err;
  }
  return res.json();
}

export async function getVoiceTrainingStatus(): Promise<{
  has_profile: boolean;
  profile?: Record<string, unknown>;
}> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/voice-training/status`, { headers });
  return res.json();
}

export async function getProjects() {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/projects/`, { headers });
  return res.json();
}

export async function createProject(name: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/projects/`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function processVocal(
  blob: Blob,
  projectName: string,
  autotuneLevel: string,
  language: string
): Promise<{
  project_id: string;
  processed_url: string;
  autotune_level: string;
  message_en: string;
  message_ar: string;
}> {
  const headers = await getAuthHeaders();
  const form = new FormData();
  form.append("file", blob, "vocal.webm");
  form.append("project_name", projectName);
  form.append("autotune_level", autotuneLevel);
  form.append("language", language);
  const res = await fetch(`${API_URL}/api/studio/process-vocal`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw err;
  }
  return res.json();
}

export async function replaceVocal(
  projectId: string,
  blob: Blob,
  autotuneLevel?: string,
  language?: string
): Promise<{
  project_id: string;
  processed_url: string;
  autotune_level: string;
  preserved_beat: boolean;
  message_en: string;
  message_ar: string;
}> {
  const headers = await getAuthHeaders();
  const form = new FormData();
  form.append("file", blob, "vocal.webm");
  if (autotuneLevel) form.append("autotune_level", autotuneLevel);
  if (language) form.append("language", language);
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/replace-vocal`, {
    method: "POST",
    headers,
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw err;
  }
  return res.json();
}

export async function deleteProject(id: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/projects/${id}`, { method: "DELETE", headers });
  if (!res.ok) await throwHttpError(res);
}

export interface BeatResult {
  beat_url: string;
  cut: number;
  cut_label: string;
  parent_cut: number | null;
  total_cuts: number;
  unlimited: boolean;
  attempt: number;
  genre: string;
  key: string;
  tempo_bpm: number;
  emotion: string;
  message_en: string;
  message_ar: string;
}

/** Generation is a JOB: submit, then poll. One fragile multi-minute HTTP
 * request used to die to browser timeouts and refreshes; polling survives
 * both, and an in-flight job can be re-attached after a refresh. */
export async function generateBeat(
  projectId: string,
  styleHint: string = "",
  branchFrom: number | null = null
): Promise<BeatResult> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/generate-beat`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({ style_hint: styleHint, branch_from: branchFrom }),
  });
  if (!res.ok) await throwHttpError(res);
  const { job_id } = await res.json();
  return pollBeatJob(projectId, job_id);
}

/** Poll a beat job until it finishes (10-minute ceiling). Transient poll
 * failures are tolerated — only a definitive failed/missing job throws. */
export async function pollBeatJob(projectId: string, jobId: string): Promise<BeatResult> {
  const deadline = Date.now() + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 3000));
    let res: Response;
    try {
      const headers = await getAuthHeaders();
      res = await fetch(`${API_URL}/api/studio/projects/${projectId}/jobs/${jobId}`, { headers });
    } catch {
      continue; // network blip — keep polling
    }
    if (res.status === 404) await throwHttpError(res);
    if (!res.ok) continue; // transient server hiccup — keep polling
    const data = await res.json();
    if (data.status === "done") return data.result as BeatResult;
    if (data.status === "failed") throw { detail: data.error };
  }
  throw {
    detail: {
      message_en: "Generation is taking unusually long. It may still finish — check your cuts in a minute.",
      message_ar: "التوليد يستغرق وقتاً أطول من المعتاد. قد يكتمل قريباً — تحقق من نسخك بعد دقيقة.",
    },
  };
}

/** The newest queued/running beat job for this project, if any — used to
 * resume the generating screen after a page refresh. */
export async function getActiveBeatJob(projectId: string): Promise<string | null> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/jobs/active?kind=beat`, { headers });
  if (!res.ok) return null;
  const data = await res.json();
  return data.job_id ?? null;
}

export async function acceptBeat(projectId: string): Promise<void> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/accept-beat`, {
    method: "POST",
    headers,
  });
  if (!res.ok) await throwHttpError(res);
}

export async function getBeatUrl(projectId: string): Promise<{
  beat_url: string;
  beat_attempts: number;
  last_genre?: string | null;
}> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/beat-url`, { headers });
  if (!res.ok) throw await res.json();
  return res.json();
}

export interface ProducerCut {
  cut: number;
  label: string;
  genre: string;
  genre_label?: string;
  key: string;
  tempo: number;
  emotion: string;
  score: number | null;
  parent_cut: number | null;
  favorite: boolean;
  created_at: string;
  beat_url?: string;
  /** The producer's note — what the AI actually heard and decided for this cut. */
  note_en?: string;
  note_ar?: string;
}

export async function listCuts(projectId: string): Promise<{
  cuts: ProducerCut[];
  total: number;
  favorites: ProducerCut[];
}> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/cuts`, { headers });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function favoriteCut(projectId: string, cut: number): Promise<{ cut: number; favorite: boolean }> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/cuts/${cut}/favorite`, {
    method: "POST",
    headers,
  });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function restoreCut(projectId: string, cut: number): Promise<{ restored: number; label: string; beat_url: string | null }> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/cuts/${cut}/restore`, {
    method: "POST",
    headers,
  });
  if (!res.ok) throw await res.json();
  return res.json();
}

export interface CoachSection {
  id: string;
  time_hint: string;
  observation: string;
  fix: string;
  should_rerecord: boolean;
  type: "strength" | "improvement" | "critical";
}

export interface CoachFeedback {
  overall_assessment: string;
  rating: number;
  strengths: string[];
  sections: CoachSection[];
  final_message: string;
}

export async function getCoachFeedback(projectId: string): Promise<CoachFeedback> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/coach-feedback`, {
    method: "POST",
    headers,
  });
  if (!res.ok) {
    const err = await res.json();
    throw err;
  }
  return res.json();
}

export async function skipCoaching(projectId: string): Promise<void> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/skip-coaching`, {
    method: "POST",
    headers,
  });
  if (!res.ok) await throwHttpError(res);
}

export async function createMix(projectId: string): Promise<{
  mp3_url: string;
  wav_url: string;
  message_en: string;
  message_ar: string;
}> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/mix`, {
    method: "POST",
    headers,
  });
  if (!res.ok) {
    const err = await res.json();
    throw err;
  }
  return res.json();
}

export async function getDownloadUrls(projectId: string): Promise<{
  mp3_url: string | null;
  wav_url: string | null;
  name: string;
}> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/download-urls`, { headers });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function submitProjectFeedback(
  projectId: string,
  ratings: {
    beat_quality: number;
    vocal_preservation: number;
    overall_satisfaction: number;
  }
): Promise<{ saved: boolean }> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/studio/projects/${projectId}/feedback`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify(ratings),
  });
  if (!res.ok) throw await res.json();
  return res.json();
}
