"""
Tests for the vocal-first reactive engine: the performance map + its wiring into
the beat synthesizer.

Covers the doctrine guarantees and the constraints that must never regress:
  - the arrangement is shaped by the performance (chorus on the real peak)
  - a flat take stays steady (no fabricated drama)
  - degenerate vocals never crash (None → template fallback)
  - the no-performance render path stays deterministic
  - the SAME take reproduces byte-for-byte ACROSS PROCESSES (crc32, not salted
    hash()) — an in-process check structurally cannot catch this, so it runs a
    subprocess with a different PYTHONHASHSEED.
"""
import io
import os
import sys
import subprocess
import hashlib

import numpy as np
import soundfile as sf

from app.services.performance_map import build_performance_map
from app.services.beat_synthesizer import generate_beat

SR = 22050
TEMPO = 100.0
_BEAT = 60.0 / TEMPO
_BAR = 4 * _BEAT

ANALYSIS = {
    "tempo": 100.0, "key": "A", "mode": "minor", "overall_rms": 0.16,
    "valence": 0.4, "emotion": "energetic", "vocal_style": "melodic",
    "density": 3.0, "swing_ratio": 0.5,
}


def _vocal(bar_amps, sung_frac=0.8, f0=220.0):
    """A fake 'vocal': one sung phrase per bar at the given amplitude, with a
    breath gap at the end of each bar. bar_amps sets the energy/tension shape."""
    out = []
    for amp in bar_amps:
        n_bar = int(SR * _BAR)
        n_sung = int(n_bar * sung_frac)
        t = np.arange(n_sung) / SR
        sig = np.zeros(n_sung, dtype=np.float32)
        for h in (1, 2, 3, 4):
            sig += (amp / h) * np.sin(2 * np.pi * f0 * h * t)
        sig += amp * 0.4 * np.sin(2 * np.pi * f0 * 6 * t)
        if n_sung:
            env = np.minimum(1, np.minimum(t * 40, (n_sung / SR - t + 1e-6) * 40))
            sig *= env
        out.append(sig.astype(np.float32))
        out.append(np.zeros(n_bar - n_sung, dtype=np.float32))
    y = np.concatenate(out) if out else np.zeros(1, dtype=np.float32)
    y = y + np.random.RandomState(0).randn(len(y)).astype(np.float32) * 0.0008
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _wav(y):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(y, np.float32), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_map_shape_is_valid_or_none():
    pm = build_performance_map(_vocal([0.2, 0.3, 0.5, 0.8, 0.9, 0.95]), TEMPO, 16)
    assert pm is not None
    assert len(pm["sections"]) == 16
    assert all(isinstance(s, str) for s in pm["sections"])
    assert len(pm["bar_tension"]) == 16
    assert all(0.0 <= t <= 1.0 for t in pm["bar_tension"])
    assert 0.0 <= pm["confidence"] <= 1.0


def test_chorus_lands_on_the_real_peak():
    # Energy builds to the end → the chorus must be in the second half.
    pm = build_performance_map(_vocal([0.12, 0.16, 0.22, 0.3, 0.5, 0.75, 0.9, 0.97]), TEMPO, 16)
    assert pm is not None
    chorus_bars = [i for i, s in enumerate(pm["sections"][:8]) if s in ("chorus", "chorus2")]
    assert chorus_bars, "expected a chorus in the build"
    assert min(chorus_bars) >= 4, f"chorus should land on the peak, got {pm['sections'][:8]}"


def test_flat_take_stays_steady():
    # A genuinely flat performance must not be given a fabricated verse/chorus swing.
    pm = build_performance_map(_vocal([0.5] * 8), TEMPO, 16)
    assert pm is not None
    assert float(np.ptp(pm["bar_tension"])) < 0.25, "flat take should have low tension swing"


def test_degenerate_inputs_never_crash():
    cases = [
        _wav(np.zeros(0)),                                  # empty
        _wav(np.random.RandomState(1).randn(int(SR * 0.1)) * 0.1),  # <0.5s
        _wav(np.zeros(int(SR * 3))),                        # pure silence
        _wav(np.random.RandomState(2).randn(int(SR * 3)) * 1e-5),   # near-silent
        _wav(np.sin(2 * np.pi * 220 * np.arange(int(SR * 2.4)) / SR) * 0.3),  # single bar
        _wav(np.sin(2 * np.pi * 220 * np.arange(int(SR * 40)) / SR) * 0.3),   # long, one pitch
        _wav(np.random.RandomState(3).randn(int(SR * 4)) * 0.2),    # white noise
    ]
    for b in cases:
        pm = build_performance_map(b, TEMPO, 16)
        assert pm is None or (len(pm["sections"]) == 16 and len(pm["bar_tension"]) == 16)
        # Must render regardless of whatever the map returned.
        wav, _ = generate_beat(analysis=ANALYSIS, bars=16, attempt=1, master=True, performance=pm)
        y, _ = sf.read(io.BytesIO(wav))
        assert y.size > 0 and not np.any(~np.isfinite(y)) and float(np.max(np.abs(y))) <= 1.0001


def test_tempo_extremes_render():
    y = np.sin(2 * np.pi * 220 * np.arange(int(SR * 8)) / SR).astype(np.float32) * 0.3
    for tempo in (1.0, 60.0, 190.0, 500.0):
        pm = build_performance_map(_wav(y), tempo, 16)
        a = {**ANALYSIS, "tempo": max(60, min(190, tempo))}
        wav, _ = generate_beat(analysis=a, bars=16, attempt=1, master=False, performance=pm)
        out, _ = sf.read(io.BytesIO(wav))
        assert not np.any(~np.isfinite(out))


def test_no_performance_path_is_deterministic():
    b1, _ = generate_beat(analysis=ANALYSIS, bars=16, attempt=3, master=False)
    b2, _ = generate_beat(analysis=ANALYSIS, bars=16, attempt=3, master=False)
    assert b1 == b2


def test_performance_render_is_valid_and_reproducible():
    pm = build_performance_map(_vocal([0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.6, 0.4]), TEMPO, 16)
    assert pm is not None
    b1, g1 = generate_beat(analysis=ANALYSIS, bars=16, attempt=2, master=False, performance=pm)
    b2, g2 = generate_beat(analysis=ANALYSIS, bars=16, attempt=2, master=False, performance=pm)
    assert b1 == b2 and g1 == g2
    y, _ = sf.read(io.BytesIO(b1))
    assert not np.any(~np.isfinite(y))


def test_different_takes_diverge():
    # Anti-fingerprint: two different performances must not yield the same beat.
    pm1 = build_performance_map(_vocal([0.2, 0.4, 0.7, 0.95, 0.5, 0.8]), TEMPO, 16)
    pm2 = build_performance_map(_vocal([0.9, 0.4, 0.2, 0.6, 0.95, 0.3]), TEMPO, 16)
    b1, _ = generate_beat(analysis=ANALYSIS, bars=16, attempt=1, master=False, performance=pm1)
    b2, _ = generate_beat(analysis=ANALYSIS, bars=16, attempt=1, master=False, performance=pm2)
    assert b1 != b2


_XPROC = """
import io, hashlib, numpy as np, soundfile as sf
from app.services.performance_map import build_performance_map
from app.services.beat_synthesizer import generate_beat
SR=22050; TEMPO=100.0; bd=60/TEMPO*4
out=[]
for a in [0.15,0.2,0.3,0.45,0.7,0.9,0.95,0.85]:
    n=int(SR*bd*0.8); t=np.arange(n)/SR
    s=sum((a/h)*np.sin(2*np.pi*220*h*t) for h in (1,2,3,4)).astype(np.float32)
    out+=[s, np.zeros(int(SR*bd*0.2), np.float32)]
buf=io.BytesIO(); sf.write(buf, np.concatenate(out), SR, format="WAV", subtype="PCM_16")
pm=build_performance_map(buf.getvalue(), TEMPO, 16)
a={"tempo":100.0,"key":"A","mode":"minor","overall_rms":0.16,"valence":0.4,
   "emotion":"energetic","vocal_style":"melodic","density":3.0,"swing_ratio":0.5}
b,_=generate_beat(analysis=a, bars=16, attempt=2, master=False, performance=pm)
print(hashlib.sha256(b).hexdigest())
"""


def _run_xproc(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed),
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", _XPROC], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_same_take_reproduces_across_processes():
    # crc32 (not PYTHONHASHSEED-salted hash()) must make the same take produce the
    # same beat across server restarts / workers. Two subprocesses, two seeds.
    h1 = _run_xproc(1)
    h2 = _run_xproc(999)
    assert h1 == h2, f"non-deterministic across processes: {h1} != {h2}"
