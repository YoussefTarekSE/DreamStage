import io
import av
import numpy as np

TARGET_SR = 22050


def load_audio(audio_bytes: bytes, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Decode audio from raw bytes (webm, opus, wav, mp4 — any format PyAV supports).
    Returns (samples_float32, sample_rate).
    """
    container = av.open(io.BytesIO(audio_bytes))

    resampler = av.AudioResampler(
        format="fltp",   # float32 planar
        layout="mono",
        rate=target_sr,
    )

    samples: list[np.ndarray] = []

    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        if resampled is None:
            continue
        frames = resampled if isinstance(resampled, list) else [resampled]
        for f in frames:
            arr = f.to_ndarray()          # shape: (1, n_samples) for mono fltp
            samples.append(arr.flatten())

    # Flush resampler
    flushed = resampler.resample(None)
    if flushed:
        flush_frames = flushed if isinstance(flushed, list) else [flushed]
        for f in flush_frames:
            samples.append(f.to_ndarray().flatten())

    if not samples:
        raise ValueError("No audio data decoded from file")

    y = np.concatenate(samples).astype(np.float32)
    return y, target_sr
