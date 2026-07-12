#!/usr/bin/env bash
# Download the real-instrument SoundFont used by the beat engine.
#
# The sample engine (app/services/sample_engine.py) renders every instrument via
# tinysoundfont — a pip package with a BUNDLED native binary — so NO system
# FluidSynth/apt package is required. It only needs a General-MIDI SoundFont file
# at backend/app/assets/soundfont.sf2.
#
# Render build command:
#   bash setup_soundfont.sh && pip install -r requirements.txt
# Local dev: run this once from the backend/ directory.
#
# GeneralUser GS — free for any use, including commercial (S. Christian Collins).
set -e

DEST="$(cd "$(dirname "$0")" && pwd)/app/assets/soundfont.sf2"
URL="https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2"

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST" 2>/dev/null)" -gt 1000000 ]; then
    echo "[setup_soundfont] SoundFont already present: $DEST"
    exit 0
fi

echo "[setup_soundfont] Downloading GeneralUser GS SoundFont (~31MB)..."
curl -fSL --retry 3 -o "$DEST" "$URL"

SIZE=$(du -h "$DEST" 2>/dev/null | cut -f1)
echo "[setup_soundfont] SoundFont installed: $DEST ($SIZE)"

# ── Real drum one-shots (Alesis SR16, MIT) — core kit for beat_synthesizer ────
DRUM_DIR="$(cd "$(dirname "$0")" && pwd)/app/assets/drums"
DRUM_BASE="https://raw.githubusercontent.com/avarasp/SR16-drum-samples-free-pack/main/free-pack"
mkdir -p "$DRUM_DIR"
for f in Kick Snare ClosedHat OpenHat Claps; do
    if [ ! -f "$DRUM_DIR/$f.wav" ]; then
        curl -fSL --retry 3 -o "$DRUM_DIR/$f.wav" "$DRUM_BASE/$f.wav" || echo "[setup] warn: $f.wav"
    fi
done
echo "[setup_soundfont] Drum samples ready in $DRUM_DIR"
