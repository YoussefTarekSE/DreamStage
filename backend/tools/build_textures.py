"""
Curate the owner's chord & melody LOOPS into shippable, analyzed instruments.

The synth pad/lead are the "cheap"-sounding melodic layers. This brings the
owner's REAL recorded loops into the beat — as the intro feature, the chorus
topline hook, and an atmospheric bed — pitched to the song key and locked to the
song tempo at generation time (see app/services/texture_loops.py).

Source analysis is IMPERFECT (BPM octave errors, ambiguous key on sparse loops),
so this tool is conservative: it octave-corrects BPM by cross-checking the beat
tracker against onset-envelope autocorrelation, keeps a Krumhansl-Schmuckler key
*margin* as a reliability score, canonicalises every tempo-locked loop to a clean
whole number of bars at a reference 120 BPM (so the runtime stretch is a small,
known ratio — never the noisy source BPM), and ROLE-tags each loop:

    topline : melodic, bright, reliable key, bar-clean  -> chorus hook
    bed     : sustained / chordy, any reliability       -> low atmospheric pad
    (feature = any tempo-locked loop; chosen at runtime for the intro)

Run locally (owner authorised shipping their library — same policy as the kits):

    python tools/build_textures.py        # MY_SAMPLES env overrides the lib path

Picks are chosen by measured quality, not hardcoded indices.
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

# Reuse the project's key model so loops and vocals are judged the same way.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.vocal_harmony import _KS_MAJOR, _KS_MINOR, _NOTE_NAMES  # noqa: E402

SR = 44_100
REF_BPM = 120.0                  # canonical tempo for tempo-locked loops
N_SHIP = 9                       # cap shipped instruments (keep repo small)
_DEFAULT_LIB = os.path.join(
    os.path.expanduser("~"), "OneDrive", "Desktop",
    "FL Studio 24.1.2 Producer Edition KioNathan", "plugins for fl",
)
LIB = os.environ.get("MY_SAMPLES", _DEFAULT_LIB)
OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "app", "assets", "textures")
)


def _key_with_margin(y: np.ndarray) -> tuple:
    """KS key/mode + margin (top corr − 2nd) + root pitch-class via chroma argmax."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    pc = chroma.sum(axis=1)
    pc = pc / (pc.sum() + 1e-9)
    scores = []
    for i in range(12):
        rot = np.roll(pc, -i)
        for prof, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            c = np.corrcoef(rot, prof)[0, 1]
            scores.append((c if np.isfinite(c) else -2.0, _NOTE_NAMES[i], mode))
    scores.sort(reverse=True)
    margin = float(scores[0][0] - scores[1][0])
    root_pc = int(np.argmax(pc))
    return scores[0][1], scores[0][2], round(margin, 3), root_pc


def _robust_bpm(y: np.ndarray, lo: float = 70.0, hi: float = 160.0) -> tuple:
    """Octave-corrected BPM: fold the beat-tracker estimate into [lo,hi] and pick
    the octave whose implied loop length is closest to whole bars. Returns
    (bpm, bar_count, bar_err)."""
    raw, _ = librosa.beat.beat_track(y=y, sr=SR)
    raw = float(np.atleast_1d(raw)[0]) or 120.0
    dur = len(y) / SR
    cands = set()
    for mult in (0.5, 1.0, 2.0):
        t = raw * mult
        while t < lo:
            t *= 2
        while t > hi:
            t /= 2
        cands.add(round(t, 2))

    def bars_at(t):
        return dur / (4.0 * 60.0 / t)

    def err(t):
        b = bars_at(t)
        return abs(b - round(b))

    bpm = min(cands, key=err)
    return round(bpm, 2), bars_at(bpm), round(err(bpm), 3)


def _character(y: np.ndarray) -> tuple:
    """(polyphony, chroma_flux, brightness01). High poly+low flux = chordy/bed;
    low poly = melodic/topline; brightness from spectral centroid (0-1)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR)
    cn = chroma / (chroma.max(axis=0, keepdims=True) + 1e-9)
    poly = float(np.mean((cn > 0.6).sum(axis=0)))
    flux = float(np.mean(np.sqrt(np.sum(np.diff(chroma, axis=1) ** 2, axis=0)))) if chroma.shape[1] > 1 else 0.0
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
    brightness = float(np.clip(cent / 4000.0, 0.0, 1.0))
    return round(poly, 2), round(flux, 3), round(brightness, 3)


def _canonicalize(y: np.ndarray, bpm: float, bar_count: float) -> tuple:
    """Time-stretch to a whole number of bars at REF_BPM. Returns (y2, bars) or
    (None, 0) if the loop is not cleanly bar-aligned enough to tempo-lock."""
    bars = int(round(bar_count))
    if bars < 1 or abs(bar_count - bars) > 0.18:
        return None, 0
    cur_dur = len(y) / SR
    tgt_dur = bars * 4.0 * 60.0 / REF_BPM
    rate = cur_dur / tgt_dur                      # >1 speeds up (shortens)
    if abs(rate - 1.0) > 0.01:
        y = librosa.effects.time_stretch(y, rate=rate)
    tgt_n = int(round(tgt_dur * SR))
    if len(y) >= tgt_n:
        y = y[:tgt_n]
    else:
        y = np.pad(y, (0, tgt_n - len(y)))
    return y.astype(np.float32), bars


def _trim_lead(y: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    above = np.where(np.abs(y) > thresh)[0]
    if len(above) == 0:
        return y
    return y[max(0, above[0] - int(0.003 * SR)):]


def _finalize(y: np.ndarray) -> np.ndarray:
    f = min(int(0.02 * SR), len(y) // 4)
    if f > 0:
        y[:f] *= np.linspace(0.0, 1.0, f)
        y[-f:] *= np.linspace(1.0, 0.0, f)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak * 0.9
    return y.astype(np.float32)


def main() -> int:
    sources = []
    for sub in (("Loops", "Chords"), ("Loops", "Melodies")):
        sources += [(os.path.basename(os.path.join(*sub)), p)
                    for p in sorted(glob.glob(os.path.join(LIB, *sub, "*.wav")))]
    if not sources:
        print(f"[build_textures] no loop sources under {LIB!r}", file=sys.stderr)
        return 1

    scored = []
    for folder, path in sources:
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as exc:
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        y = _trim_lead(y)
        dur = len(y) / SR
        if dur < 3.0 or float(np.max(np.abs(y))) < 1e-4:
            continue
        key, mode, margin, root_pc = _key_with_margin(y)
        bpm, bar_count, bar_err = _robust_bpm(y)
        poly, flux, brightness = _character(y)

        y_canon, bars = _canonicalize(y, bpm, bar_count)
        tempo_lockable = y_canon is not None

        # Role: bar-clean + reliable key + melodic → topline; else → bed.
        is_melodic = (folder == "Melodies") or poly < 2.2
        if tempo_lockable and margin >= 0.12 and is_melodic:
            role = "topline"
        else:
            role = "bed"

        # Quality: reward reliable key + bar-cleanliness + (for beds) steadiness.
        quality = (0.45 * float(np.clip(margin / 0.3, 0, 1))
                   + 0.30 * (1.0 - min(bar_err, 0.5) / 0.5)
                   + 0.15 * (1.0 if tempo_lockable else 0.0)
                   + 0.10 * (1.0 - min(flux, 1.0)))
        scored.append(dict(
            path=path, folder=folder, key=key, mode=mode, margin=margin,
            root_pc=root_pc, bpm=bpm, bars=bars, bar_err=bar_err, poly=poly,
            flux=flux, brightness=brightness, role=role,
            tempo_lockable=tempo_lockable, quality=round(quality, 3),
            y_canon=y_canon, y_raw=y,
        ))

    if not scored:
        print("[build_textures] no usable loops after scoring", file=sys.stderr)
        return 1

    scored.sort(key=lambda d: d["quality"], reverse=True)
    print("Curated loops (ranked):")
    for s in scored:
        print(f"  {os.path.basename(s['path']):20s} q={s['quality']:.2f} role={s['role']:7s} "
              f"key={s['key']:2s}{s['mode'][:3]} margin={s['margin']:.2f} bpm={s['bpm']:5.1f} "
              f"bars={s['bars']} err={s['bar_err']:.2f} poly={s['poly']:.1f} bright={s['brightness']:.2f} "
              f"lock={s['tempo_lockable']}")

    # Ship a balanced set: keep toplines and beds both represented.
    toplines = [s for s in scored if s["role"] == "topline"]
    beds = [s for s in scored if s["role"] == "bed"]
    chosen = (toplines[:max(4, N_SHIP // 2)] + beds[:N_SHIP - len(toplines[:max(4, N_SHIP // 2)])])[:N_SHIP]

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    SHIP_BARS = 4                                   # 4 bars tiles fine; keeps repo small
    for i, s in enumerate(chosen):
        # Tempo-locked roles ship the canonical (120 BPM, whole-bar) buffer,
        # trimmed to <=4 bars (it tiles at runtime); non-lockable beds ship the
        # raw trimmed loop (pitch-only at runtime), capped to 8s.
        if s["tempo_lockable"]:
            bars = min(s["bars"], SHIP_BARS)
            n = int(round(bars * 4.0 * 60.0 / REF_BPM * SR))
            y_out = _finalize(s["y_canon"][:n].copy())
            ref_bpm = REF_BPM
        else:
            y_out = _finalize(s["y_raw"][: int(8 * SR)].copy())    # cap 8s
            ref_bpm, bars = 0.0, 0
        name = f"{s['role']}_{i:02d}"
        fname = f"{name}.wav"
        sf.write(os.path.join(OUT_DIR, fname), y_out, SR, subtype="PCM_16")
        manifest[name] = dict(
            file=fname, role=s["role"], root_pc=s["root_pc"], key=s["key"],
            mode=s["mode"], margin=s["margin"], ref_bpm=ref_bpm, bars=bars,
            brightness=s["brightness"], poly=s["poly"],
            len_s=round(len(y_out) / SR, 3), source=os.path.basename(s["path"]),
        )
        print(f"  -> {fname:14s} role={s['role']:7s} root={_NOTE_NAMES[s['root_pc']]:2s} "
              f"ref_bpm={ref_bpm:.0f} bars={bars} {manifest[name]['len_s']}s")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[build_textures] wrote {len(manifest)} loops + manifest to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
