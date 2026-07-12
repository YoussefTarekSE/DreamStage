"""
Build shippable drum kits from the owner's sample library into
app/assets/drums/kits/<kit>/<role>.wav (committed + deployed — the owner holds
the rights and authorised redistribution).

Picks are analysis-driven where names are generic: kicks ranked by low-end
weight (boomiest -> trap), snares by brightness, plus sensible name matches for
hats/claps. Safe loader handles plain PCM and Ogg-in-WAV without segfaulting.

Run:  python tools/build_kits.py
Override source:  set MY_SAMPLES="<path to library>"
"""
import io
import os
import struct
import glob
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

LIB = os.environ.get(
    "MY_SAMPLES",
    r"C:/Users/youss/OneDrive/Desktop/FL Studio 24.1.2 Producer Edition KioNathan/plugins for fl",
)
TW = os.path.join(LIB, "TW's Drum Essentials V1")
OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "app", "assets", "drums", "kits"))


def safe_load(path):
    raw = open(path, "rb").read()
    data = None
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        pos = 12
        while pos + 8 <= len(raw):
            cid = raw[pos:pos + 4]
            sz = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
            if cid == b"data":
                data = raw[pos + 8:pos + 8 + sz]
                break
            pos += 8 + sz + (sz & 1)
    if data is not None and data[:4] == b"OggS":
        y, sr = sf.read(io.BytesIO(data))
    else:
        sr, y = wavfile.read(path)
        y = np.asarray(y, dtype=np.float64)
        if np.issubdtype(np.asarray(y).dtype, np.integer):
            y = y / 32768.0
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def low_energy(y, sr):
    sos = butter(4, 120 / (sr / 2), btype="low", output="sos")
    return float(np.sqrt(np.mean(sosfilt(sos, y) ** 2)))


def centroid(y, sr):
    import librosa
    return float(np.mean(librosa.feature.spectral_centroid(y=y.astype(np.float32), sr=sr)))


def write_norm(y, sr, path):
    import librosa
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100).astype(np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(path, (y / peak * 0.9).astype(np.float32), 44100, subtype="PCM_16")


def pick(globpat, key=None, reverse=False, name_contains=None):
    files = glob.glob(globpat)
    cands = []
    for f in files:
        try:
            y, sr = safe_load(f)
            cands.append((f, y, sr))
        except Exception:
            pass
    if not cands:
        return None
    if name_contains:
        named = [c for c in cands if name_contains.lower() in os.path.basename(c[0]).lower()]
        if named:
            cands = named
    if key:
        cands.sort(key=lambda c: key(c[1], c[2]), reverse=reverse)
    return cands


def main():
    kicks = pick(os.path.join(TW, "Kicks", "*.wav"), key=low_energy, reverse=True) or []
    snares = pick(os.path.join(TW, "Snares", "*.wav"), key=centroid, reverse=True) or []
    # kits: (kit, kick_index_from_boomiest, snare_index_from_brightest, clap_name, ch_name, oh_name)
    plan = {
        "trap":     dict(kick=0,  snare=0,  clap="Classic",      ch="hi hat 1", oh="open hi hat 1"),
        "dance":    dict(kick=min(3, len(kicks) - 1), snare=min(2, len(snares) - 1),
                         clap="house", ch="hi hat 2", oh="open hi hat 2"),
        "acoustic": dict(kick=None, snare=min(len(snares) - 1, 6),
                         clap="Bonk", ch="hi hat 1", oh="open hi hat 1"),
    }
    hats = os.path.join(TW, "Hi-Hats"); claps = os.path.join(TW, "Claps")
    for kit, p in plan.items():
        os.makedirs(os.path.join(OUT, kit), exist_ok=True)
        # kick
        if p["kick"] is None:  # acoustic -> prefer a 'hip hop kick' by name
            hk = [c for c in kicks if "hip hop" in os.path.basename(c[0]).lower()]
            k = hk[0] if hk else kicks[len(kicks) // 2]
        else:
            k = kicks[min(p["kick"], len(kicks) - 1)]
        write_norm(k[1], k[2], os.path.join(OUT, kit, "kick.wav"))
        s = snares[min(p["snare"], len(snares) - 1)]
        write_norm(s[1], s[2], os.path.join(OUT, kit, "snare.wav"))
        for role, sub, nm in [("closedhat", hats, p["ch"]), ("openhat", hats, p["oh"]),
                              ("clap", claps, p["clap"])]:
            cands = pick(os.path.join(sub, "*.wav"), name_contains=nm)
            if cands:
                y, sr = cands[0][1], cands[0][2]
                write_norm(y, sr, os.path.join(OUT, kit, f"{role}.wav"))
        print(f"  {kit}: kick={os.path.basename(k[0])}  snare={os.path.basename(s[0])}")
    print(f"[build_kits] kits written -> {OUT}")


if __name__ == "__main__":
    main()
