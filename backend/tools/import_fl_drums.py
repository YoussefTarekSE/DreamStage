"""
Local setup tool: import curated drum one-shots from a local FL Studio install
into app/assets/drums/fl/<kit>/ as clean PCM WAVs the beat engine can use.

FL stores most samples as Ogg-Vorbis-INSIDE-WAV, which segfaults soundfile/PyAV.
We decode safely by extracting the OGG stream from the WAV `data` chunk and
reading it as standalone Ogg (libsndfile decodes Ogg fine); PCM files are read
directly. NOTHING here is committed — FL samples are licensed for the owner's
productions, so app/assets/drums/fl/ is gitignored and never shipped. The
deployed product uses the MIT SR16 kit + synthesis.

Run:  python tools/import_fl_drums.py
Override FL path:  set FL_PACKS=...   (default: FL Studio 2024 packs)
"""
import io
import os
import struct
import numpy as np
import soundfile as sf

FL = os.environ.get(
    "FL_PACKS",
    r"C:/Program Files/Image-Line/FL Studio 2024/Data/Patches/Packs",
)
OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "app", "assets", "drums", "fl"))

# Genre kits → {role: relative path under the FL packs dir}
KITS = {
    "trap": {  # trap/drill/phonk/modern hip-hop — the 808 machine
        "kick":      "Drums/Kicks/808 Kick.wav",
        "snare":     "Drums/Snares/808 Snare.wav",
        "closedhat": "Drums/Hats/808 CH.wav",
        "openhat":   "Drums/Hats/808 OH.wav",
        "clap":      "Legacy/Drums/Dance/Basic 808 Clap.wav",
    },
    "dance": {  # pop/house/afrobeats/dancehall — the 909
        "kick":      "Drums/Kicks/909 Kick.wav",
        "snare":     "Drums/Snares/909 Snare.wav",
        "closedhat": "Drums/Hats/909 CH 1.wav",
        "openhat":   "Drums/Hats/909 OH.wav",
        "clap":      "Legacy/Drums/Dance/DNC_Clap.wav",
    },
    "acoustic": {  # rnb/soul/lofi/boom-bap — real-kit feel
        "kick":      "Legacy/Drums/HipHop/HIP_Kick.wav",
        "snare":     "Legacy/Drums/HipHop/HIP_Snare.wav",
        "closedhat": "Legacy/Drums/HipHop/HIP_Hat.wav",
        "openhat":   "Drums/Hats/909 OH.wav",
        "clap":      "Legacy/Drums/Dance/Basic 808 Clap.wav",
    },
}


def decode_fl(path: str) -> tuple:
    """Return (mono float32, sr). Handles Ogg-in-WAV and plain PCM safely."""
    raw = open(path, "rb").read()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    pos, data = 12, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        sz = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"data":
            data = raw[pos + 8:pos + 8 + sz]
            break
        pos += 8 + sz + (sz & 1)
    if data is not None and data[:4] == b"OggS":          # Ogg-Vorbis-in-WAV
        y, sr = sf.read(io.BytesIO(data))
    else:                                                  # plain PCM
        from scipy.io import wavfile
        sr, y = wavfile.read(path)
        y = y.astype(np.float64)
        if np.issubdtype(np.asarray(y).dtype, np.integer):
            y = y / 32768.0
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def main():
    if not os.path.isdir(FL):
        print(f"[import_fl_drums] FL packs not found at {FL} — skipping.")
        return
    import librosa
    ok = bad = 0
    for kit, roles in KITS.items():
        os.makedirs(os.path.join(OUT, kit), exist_ok=True)
        for role, rel in roles.items():
            src = os.path.join(FL, rel)
            try:
                y, sr = decode_fl(src)
                if sr != 44100:
                    y = librosa.resample(y, orig_sr=sr, target_sr=44100).astype(np.float32)
                peak = float(np.max(np.abs(y))) or 1.0
                y = (y / peak * 0.9).astype(np.float32)
                sf.write(os.path.join(OUT, kit, f"{role}.wav"), y, 44100, subtype="PCM_16")
                ok += 1
            except Exception as e:
                print(f"  skip {kit}/{role} ({rel}): {type(e).__name__} {str(e)[:60]}")
                bad += 1
    print(f"[import_fl_drums] done: {ok} imported, {bad} skipped -> {OUT}")


if __name__ == "__main__":
    main()
