"""
Local MusicGen inference via Facebook's audiocraft library.

This is the gold-standard beat generation path: conditioned on the actual
vocal melody (not just a text description of it). MusicGen's melody
conditioning works by projecting the input audio into chroma features and
using cross-attention in the transformer to steer generation.

Paper:
  Copet et al. (2023) "Simple and Controllable Music Generation"
  NeurIPS 2023. https://arxiv.org/abs/2306.05284

Repository: https://github.com/facebookresearch/audiocraft  (MIT license)

Model variants and compute requirements:
  ┌────────────────────┬────────────┬─────────┬───────────┬─────────────┐
  │ Model              │ Parameters │ RAM     │ GPU time  │ CPU time    │
  ├────────────────────┼────────────┼─────────┼───────────┼─────────────┤
  │ musicgen-small     │  300 M     │  2 GB   │  ~20 s    │  ~3–5 min   │
  │ musicgen-medium    │  1.5 B     │  6 GB   │  ~50 s    │  Not viable │
  │ musicgen-large     │  3.3 B     │ 12 GB   │  ~2 min   │  Not viable │
  │ musicgen-stereo-   │  1.5 B     │  7 GB   │  ~50 s    │  Not viable │
  │   medium (stereo)  │            │         │           │             │
  └────────────────────┴────────────┴─────────┴───────────┴─────────────┘

For DreamStage at $0 budget:
  - Development: musicgen-small on CPU is slow but works.
  - Free GPU:    Use Google Colab (T4 GPU, free tier) for batch generation.
  - Production:  Use musicgen_hf.py (HF Inference API) — no GPU needed.

Melody conditioning vs text-only:
  - Text-only: model generates music matching the description
  - Melody conditioning: model generates music that follows the vocal
    melody structure while still matching the text description.
    The vocal is NOT included in the output — only its melodic contour
    influences the generated beat.

Installation:
  pip install audiocraft
  (includes torch, torchaudio, encodec)

  Verify: python -c "from audiocraft.models import MusicGen; print('OK')"
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torchaudio
    from audiocraft.models import MusicGen
    _HAS_AUDIOCRAFT = True
except ImportError:
    _HAS_AUDIOCRAFT = False
    logger.info("audiocraft not installed — local MusicGen unavailable")


# ── Model singleton (lazy load) ───────────────────────────────────────────────

_MODEL: Optional[object] = None
_MODEL_NAME: str = ""


def _get_model(model_name: str = "facebook/musicgen-small"):
    """
    Load MusicGen model once and cache globally.
    Subsequent calls return the cached model immediately.
    """
    global _MODEL, _MODEL_NAME

    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL

    if not _HAS_AUDIOCRAFT:
        raise ImportError(
            "audiocraft is required for local MusicGen.\n"
            "Install: pip install audiocraft\n"
            "Note: requires ~2 GB disk, ~2 GB RAM (musicgen-small)"
        )

    logger.info("Loading MusicGen model: %s (first load is slow)", model_name)
    model = MusicGen.get_pretrained(model_name)
    _MODEL = model
    _MODEL_NAME = model_name
    logger.info("MusicGen model loaded")
    return model


# ── Core generation function ──────────────────────────────────────────────────

def generate_with_melody_conditioning(
    prompt: str,
    vocal_array: np.ndarray,
    vocal_sr: int,
    duration_sec: float = 30.0,
    model_name: str = "facebook/musicgen-small",
    guidance_scale: float = 3.0,
    top_k: int = 250,
) -> bytes:
    """
    Generate a beat conditioned on vocal melody + text description.

    The vocal_array's chroma features (pitch class distribution over time)
    are used as a conditioning signal. The generated beat will follow the
    harmonic/melodic structure of the vocal while sounding like the genre
    described in the prompt.

    Args:
        prompt:        Text description ("melodic trap instrumental, 140 BPM, key of A minor...")
        vocal_array:   Mono audio array of the processed vocal
        vocal_sr:      Sample rate of vocal_array
        duration_sec:  Length of generated beat in seconds
        model_name:    HuggingFace model ID for MusicGen variant
        guidance_scale: Classifier-free guidance scale (higher = follows prompt more strictly)
        top_k:         Sampling top-k for diversity (lower = more conservative)

    Returns:
        WAV audio bytes (mono, 32 kHz)

    Raises:
        ImportError:   if audiocraft is not installed
        RuntimeError:  if generation fails
    """
    model = _get_model(model_name)
    model.set_generation_params(
        duration=duration_sec,
        guidance_scale=guidance_scale,
        top_k=top_k,
    )

    # Convert vocal to torch tensor for melody conditioning
    # torchaudio.load expects a file; encode to tensor manually
    vocal_tensor = torch.tensor(vocal_array, dtype=torch.float32)
    if vocal_tensor.ndim == 1:
        vocal_tensor = vocal_tensor.unsqueeze(0)  # (1, samples)
    vocal_tensor = vocal_tensor.unsqueeze(0)       # (1, 1, samples)

    # Resample vocal to model's expected sample rate (32 kHz) if needed
    model_sr = model.sample_rate
    if vocal_sr != model_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=vocal_sr, new_freq=model_sr
        )
        vocal_tensor = resampler(vocal_tensor.squeeze(0)).unsqueeze(0)

    logger.info("Generating beat: prompt='%s...', duration=%.0fs, model=%s",
                prompt[:60], duration_sec, model_name)

    with torch.no_grad():
        generated = model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=vocal_tensor,
            melody_sample_rate=model_sr,
            progress=False,
        )

    # Convert output tensor (1, channels, samples) → WAV bytes
    audio = generated[0]   # (channels, samples)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # mix to mono

    buf = io.BytesIO()
    torchaudio.save(buf, audio.cpu(), model_sr, format="wav")
    buf.seek(0)
    return buf.read()


def generate_text_only(
    prompt: str,
    duration_sec: float = 30.0,
    model_name: str = "facebook/musicgen-small",
) -> bytes:
    """
    Generate a beat from text description only (no melody conditioning).
    Faster than melody conditioning; useful for quick previews.
    """
    model = _get_model(model_name)
    model.set_generation_params(duration=duration_sec)

    with torch.no_grad():
        generated = model.generate(descriptions=[prompt], progress=False)

    audio = generated[0]
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    buf = io.BytesIO()
    torchaudio.save(buf, audio.cpu(), model.sample_rate, format="wav")
    buf.seek(0)
    return buf.read()


def is_available() -> bool:
    """Check if local MusicGen is available and loadable."""
    return _HAS_AUDIOCRAFT


def get_info() -> dict:
    """Return information about the local MusicGen setup."""
    if not _HAS_AUDIOCRAFT:
        return {
            "available": False,
            "reason": "audiocraft not installed",
            "install": "pip install audiocraft",
        }
    try:
        import torch
        return {
            "available": True,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "recommended_model": (
                "facebook/musicgen-small"
                if not torch.cuda.is_available()
                else "facebook/musicgen-medium"
            ),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ── CLI for testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Test local MusicGen")
    parser.add_argument("--prompt", default="melodic trap instrumental, 140 BPM, key of A minor, deep 808 bass, crisp hi-hats, no vocals")
    parser.add_argument("--vocal", default=None, help="Path to vocal WAV for melody conditioning")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", default="test_beat.wav")
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    if args.info:
        import json
        print(json.dumps(get_info(), indent=2))
        exit(0)

    if not is_available():
        print("audiocraft not installed. Run: pip install audiocraft")
        exit(1)

    if args.vocal:
        print(f"Generating with melody conditioning from: {args.vocal}")
        y, sr = sf.read(args.vocal)
        if y.ndim > 1:
            y = y.mean(axis=1)
        audio_bytes = generate_with_melody_conditioning(
            args.prompt, y.astype(np.float32), sr,
            duration_sec=args.duration, model_name=args.model,
        )
    else:
        print("Generating text-only...")
        audio_bytes = generate_text_only(
            args.prompt, duration_sec=args.duration, model_name=args.model
        )

    with open(args.output, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved: {args.output}")
