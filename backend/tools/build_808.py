"""
Build a shippable, tuned 808 bass instrument from the owner's Bass Shots.

The synthesized / GM-soundfont bass is the most prominent "fake/cheap"-sounding
element of a beat. An 808 is, by definition, ONE sustained tonal sample played
back at different speeds to hit different notes — exactly what a sampler does.
This script curates the best sustained, sub-heavy, tonal bass one-shots from the
owner's library, trims + micro-tunes them so their fundamental sits exactly on a
semitone, and writes them (plus a manifest of exact source pitches) to
``app/assets/bass/`` so ``bass_samples.py`` can retune them per bassline note.

Run locally (the owner authorised shipping their library — same policy as the
drum kits):

    python tools/build_808.py
    # MY_SAMPLES env var overrides the source library path.

Picks are chosen by measured quality, not by hardcoded indices, so the script
still works if the library changes.
"""
import os
import sys
import glob
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import librosa
import soundfile as sf

SR = 44_100
MAX_LEN_S = 4.0          # plenty of body for any pitched note; keeps ship size small
N_VARIANTS = 2           # primary "808" + one "808_alt"

_DEFAULT_LIB = os.path.join(
    os.path.expanduser("~"),
    "OneDrive", "Desktop",
    "FL Studio 24.1.2 Producer Edition KioNathan", "plugins for fl",
)
LIB = os.environ.get("MY_SAMPLES", _DEFAULT_LIB)
OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "app", "assets", "bass")
)

_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def _midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def _note_name(f: float) -> str:
    m = int(round(_hz_to_midi(f)))
    return f"{_NOTE[m % 12]}{m // 12 - 1}"


def _trim_lead(y: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """Drop leading silence but KEEP the attack transient (the 808 punch)."""
    above = np.where(np.abs(y) > thresh)[0]
    if len(above) == 0:
        return y
    start = max(0, above[0] - int(0.002 * SR))   # keep 2 ms pre-attack
    return y[start:]


def _detect_f0(y: np.ndarray) -> float:
    """Robust fundamental via pyin median over the stable body (post-attack)."""
    body = y[int(0.04 * SR):]
    f0, _, _ = librosa.pyin(body, fmin=22, fmax=400, sr=SR, frame_length=4096)
    f0v = f0[~np.isnan(f0)]
    return float(np.median(f0v)) if len(f0v) else 0.0


def _score(y: np.ndarray, sr: int) -> dict:
    """Quality score for an 808 source: tonal, sub-heavy, sustained, long enough."""
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-4 or dur < 0.6:
        return {"ok": False}
    f0 = _detect_f0(y)
    if not (25.0 <= f0 <= 120.0):        # must be a real low-bass fundamental
        return {"ok": False}
    # low-end energy ratio (<120 Hz)
    w = y * np.hanning(len(y))
    S = np.abs(np.fft.rfft(w))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    low = S[(fr >= 20) & (fr < 120)].sum()
    tot = S[fr < 8000].sum() + 1e-9
    low_ratio = float(low / tot)
    # sustain: tail energy vs head energy
    head = np.sqrt(np.mean(y[:int(0.1 * sr)] ** 2) + 1e-12)
    tail = (np.sqrt(np.mean(y[int(0.4 * sr):int(0.6 * sr)] ** 2) + 1e-12)
            if dur > 0.65 else 0.0)
    sustain = float(np.clip(tail / (head + 1e-9), 0, 3)) / 3.0
    # tonality: harmonic peak prominence at f0
    quality = 0.45 * low_ratio + 0.35 * sustain + 0.20 * min(dur / 3.0, 1.0)
    return {"ok": True, "f0": f0, "low_ratio": low_ratio,
            "sustain": sustain, "dur": dur, "quality": quality}


def _score_pluck(y: np.ndarray, sr: int) -> dict:
    """Quality score for a PLUCK/finger bass: fast attack, short decay, tonal, a
    low-bass fundamental with some mid presence (not a pure sustained sub)."""
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-4 or dur < 0.15 or dur > 3.5:
        return {"ok": False}
    env = np.abs(y)
    peak_i = int(np.argmax(env))
    attack = peak_i / sr
    # Plucks are short — detect f0 around the peak, not the (silent) tail.
    seg = y[peak_i: peak_i + int(0.2 * sr)]
    f0 = 0.0
    if len(seg) > 1024:
        f0a, _, _ = librosa.pyin(seg, fmin=30, fmax=180, sr=sr, frame_length=2048)
        v = f0a[~np.isnan(f0a)]
        f0 = float(np.median(v)) if len(v) else 0.0
    if not (30.0 <= f0 <= 150.0):
        return {"ok": False}
    head = np.sqrt(np.mean(env[: int(0.06 * sr)] ** 2) + 1e-12)
    tail = (np.sqrt(np.mean(y[int(0.35 * sr): int(0.55 * sr)] ** 2) + 1e-12)
            if dur > 0.6 else 0.0)
    decay = float(tail / (head + 1e-9))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    fast_attack = 1.0 - min(attack / 0.06, 1.0)      # reward < 60 ms attack
    short_decay = 1.0 - min(decay / 0.5, 1.0)        # reward a quick decay
    bright = float(np.clip(cent / 2500.0, 0, 1))     # mid presence, not pure sub
    quality = 0.40 * fast_attack + 0.40 * short_decay + 0.20 * bright
    return {"ok": True, "f0": f0, "attack": attack, "decay": decay,
            "cent": cent, "quality": round(quality, 3)}


def _micro_tune(y: np.ndarray, f0: float) -> tuple:
    """Resample so f0 lands EXACTLY on the nearest semitone (removes detection
    error → rendered notes are perfectly in tune). Returns (y_tuned, exact_hz)."""
    target_midi = round(_hz_to_midi(f0))
    target_hz = _midi_to_hz(target_midi)
    ratio = target_hz / f0                       # >1 = pitch up = play faster
    new_n = max(8, int(len(y) / ratio))
    idx = np.linspace(0, len(y) - 1, new_n)
    y2 = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return y2, target_hz


def _finalize(y: np.ndarray) -> np.ndarray:
    y = y[: int(MAX_LEN_S * SR)]
    # gentle end fade so a clipped tail never clicks
    f = min(int(0.03 * SR), len(y) // 4)
    if f > 0:
        y[-f:] *= np.linspace(1.0, 0.0, f)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak * 0.9
    return y.astype(np.float32)


def main() -> int:
    # ONLY one-shots: a retunable 808 must be a single sustained note. Loops
    # contain a progression, so pitch-shifting them would impose a fixed
    # bassline that fights the vocal-derived harmony.
    sources = sorted(glob.glob(os.path.join(LIB, "One Shots", "Bass Shots", "*.wav")))
    if not sources:
        print(f"[build_808] no bass sources found under {LIB!r}", file=sys.stderr)
        return 1

    scored = []
    for path in sources:
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        except Exception as exc:
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        y = _trim_lead(y)
        s = _score(y, SR)
        if not s.get("ok"):
            continue
        s["path"] = path
        s["y"] = y
        scored.append(s)

    if not scored:
        print("[build_808] no usable 808 sources after scoring", file=sys.stderr)
        return 1

    scored.sort(key=lambda d: d["quality"], reverse=True)
    print("Top 808 candidates:")
    for s in scored[:6]:
        print(f"  {os.path.basename(s['path']):26s} q={s['quality']:.3f} "
              f"f0={s['f0']:5.1f}Hz {_note_name(s['f0']):4s} "
              f"low={s['low_ratio']:.2f} sus={s['sustain']:.2f} dur={s['dur']:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    names = ["808"] + [f"808_alt{i}" if i > 1 else "808_alt" for i in range(1, N_VARIANTS)]
    for name, s in zip(names, scored[:N_VARIANTS]):
        y_tuned, _ = _micro_tune(s["y"], s["f0"])
        y_final = _finalize(y_tuned)
        out_path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(out_path, y_final, SR, subtype="PCM_16")
        # Re-measure the EXACT fundamental of the final file (round-trip through
        # resample + PCM16 quantisation shifts it a few cents); store the truth
        # so the renderer retunes from the real pitch and locks to the vocal key.
        measured = _detect_f0(sf.read(out_path)[0].astype(np.float32))
        if not (20.0 <= measured <= 130.0):
            measured = _midi_to_hz(round(_hz_to_midi(s["f0"])))
        manifest[name] = {
            "file": f"{name}.wav",
            "f0_hz": round(measured, 4),
            "midi": int(round(_hz_to_midi(measured))),
            "note": _note_name(measured),
            "source": os.path.basename(s["path"]),
            "len_s": round(len(y_final) / SR, 3),
        }
        print(f"  -> wrote {name}.wav  measured f0 {measured:.2f}Hz "
              f"({manifest[name]['note']})  {manifest[name]['len_s']}s")

    # ── Pluck / finger bass for the non-808 (pluck-style) genres ──────────────
    pluck_scored = []
    for path in sources:
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception:
            continue
        y = _trim_lead(y)
        ps = _score_pluck(y, SR)
        if ps.get("ok"):
            ps["path"] = path
            ps["y"] = y
            pluck_scored.append(ps)
    pluck_scored.sort(key=lambda d: d["quality"], reverse=True)
    if pluck_scored:
        print("Top pluck candidates:")
        for s in pluck_scored[:5]:
            print(f"  {os.path.basename(s['path']):26s} q={s['quality']:.3f} "
                  f"f0={s['f0']:5.1f}Hz {_note_name(s['f0']):4s} "
                  f"atk={s['attack']*1000:4.0f}ms decay={s['decay']:.2f} cent={s['cent']:.0f}Hz")
        s = pluck_scored[0]
        # Tune to the nearest semitone; trust the tuned target (short plucks
        # re-measure unreliably). Keep the natural attack/decay intact.
        y_tuned, exact_hz = _micro_tune(s["y"], s["f0"])
        y_final = _finalize(y_tuned)
        sf.write(os.path.join(OUT_DIR, "pluck.wav"), y_final, SR, subtype="PCM_16")
        manifest["pluck"] = {
            "file": "pluck.wav",
            "f0_hz": round(exact_hz, 4),
            "midi": int(round(_hz_to_midi(exact_hz))),
            "note": _note_name(exact_hz),
            "source": os.path.basename(s["path"]),
            "len_s": round(len(y_final) / SR, 3),
        }
        print(f"  -> wrote pluck.wav  tuned to {manifest['pluck']['note']} "
              f"({exact_hz:.2f}Hz)  {manifest['pluck']['len_s']}s")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[build_808] wrote {len(manifest)} instrument(s) + manifest to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
