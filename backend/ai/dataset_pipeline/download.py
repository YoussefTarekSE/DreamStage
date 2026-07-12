"""
Automated dataset downloader for DreamStage AI training pipeline.

Datasets downloaded:
  GTZAN (genre recognition)
    - 1,000 audio clips × 30 s, 10 genres, 22.05 kHz WAV
    - 1.2 GB total
    - Source: Marsyas project, widely used in MIR research
    - License: Research use

  FMA-small (Free Music Archive, small subset)
    - 8,000 tracks × 30 s, 8 genres (balanced), 22.05 kHz MP3
    - 7.2 GB total
    - Source: Defferrard et al., ISMIR 2017
    - License: Creative Commons (CC-BY, CC-BY-SA, CC-BY-NC, CC0)
    - Full metadata CSV included (mood tags, genre, BPM, key)

  NSynth (note synthesis dataset — for instrument embeddings)
    - 305,979 audio samples of 1,006 musical instruments
    - Each sample 4 s at 16 kHz
    - Source: Engel et al., ICML 2017 (Google Magenta)
    - License: Creative Commons Attribution 4.0

  GiantMIDI-Piano (optional — piano MIDI for music theory patterns)
    - 10,854 MIDI files of real piano performances
    - Source: Kong et al., 2020
    - License: CC BY 4.0
    - Only needed for training music theory / arrangement models

Usage:
    python -m backend.ai.dataset_pipeline.download --datasets gtzan fma_small
    python -m backend.ai.dataset_pipeline.download --datasets all
    python -m backend.ai.dataset_pipeline.download --list

Set DATA_DIR environment variable to change download location.
Default: backend/ai/data/
"""
from __future__ import annotations

import os
import io
import sys
import gzip
import shutil
import hashlib
import tarfile
import zipfile
import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "gtzan": {
        "description": "GTZAN genre collection — 1000 clips, 10 genres, 30s each",
        "size_gb": 1.2,
        "url": "http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        "mirror": "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
        "filename": "genres.tar.gz",
        "extract_to": "gtzan",
        "checksum_md5": None,  # varies by mirror
        "type": "tar.gz",
        "notes": "Standard genre benchmark. 100 clips per genre.",
    },
    "fma_metadata": {
        "description": "FMA metadata CSV files (no audio — just track info, mood, genre, BPM)",
        "size_gb": 0.02,
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "filename": "fma_metadata.zip",
        "extract_to": "fma_metadata",
        "type": "zip",
        "notes": "Download this first to see which FMA tracks to grab.",
    },
    "fma_small": {
        "description": "FMA-small — 8000 tracks × 30s, 8 balanced genres",
        "size_gb": 7.2,
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "filename": "fma_small.zip",
        "extract_to": "fma_small",
        "type": "zip",
        "notes": "Requires fma_metadata first for labels.",
    },
    "nsynth_train": {
        "description": "NSynth train split — 289k samples of 1006 instruments",
        "size_gb": 22.0,
        "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
        "filename": "nsynth-train.jsonwav.tar.gz",
        "extract_to": "nsynth/train",
        "type": "tar.gz",
        "notes": "Large. Only needed for timbre/instrument classifier.",
    },
    "nsynth_valid": {
        "description": "NSynth validation split — 12k samples",
        "size_gb": 0.98,
        "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
        "filename": "nsynth-valid.jsonwav.tar.gz",
        "extract_to": "nsynth/valid",
        "type": "tar.gz",
        "notes": "Smaller. Good for quick instrument embedding tests.",
    },
    "gtzan_fault_filtered": {
        "description": "GTZAN with known-faulty tracks removed (Sturm, 2013)",
        "size_gb": 1.1,
        "url": "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/gtzan_fault_filtered.zip",
        "filename": "gtzan_fault_filtered.zip",
        "extract_to": "gtzan_clean",
        "type": "zip",
        "notes": "Preferred over raw GTZAN — removes duplicates and corrupted files.",
    },
}


# ── Progress reporting ────────────────────────────────────────────────────────

def _progress_hook(filename: str) -> Generator:
    """Yield a urllib reporthook closure that prints download progress."""
    seen_blocks = [0]

    def hook(block_count: int, block_size: int, total_size: int):
        seen_blocks[0] = block_count
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f"\r  {filename}: {pct:5.1f}%  ({mb_done:.0f}/{mb_total:.0f} MB)")
            sys.stdout.flush()
        else:
            mb = block_count * block_size / 1024 / 1024
            sys.stdout.write(f"\r  {filename}: {mb:.1f} MB downloaded")
            sys.stdout.flush()

    return hook


# ── Download + extract ────────────────────────────────────────────────────────

def download_dataset(name: str, force: bool = False) -> Path:
    """
    Download and extract a registered dataset.

    Args:
        name:  Dataset key from DATASETS registry
        force: Re-download even if already present

    Returns:
        Path to the extracted dataset directory
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")

    info = DATASETS[name]
    dest_dir = DATA_DIR / info["extract_to"]

    if dest_dir.exists() and not force:
        print(f"  {name} already present at {dest_dir}")
        return dest_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / info["filename"]

    # Download
    url = info["url"]
    print(f"\nDownloading {name} ({info['size_gb']} GB)...")
    print(f"  Source: {url}")
    try:
        urllib.request.urlretrieve(url, archive_path, _progress_hook(info["filename"]))
    except Exception as primary_err:
        mirror = info.get("mirror")
        if mirror:
            print(f"\n  Primary URL failed ({primary_err}), trying mirror...")
            urllib.request.urlretrieve(mirror, archive_path, _progress_hook(info["filename"]))
        else:
            raise
    print()  # newline after progress

    # Optional checksum
    checksum = info.get("checksum_md5")
    if checksum:
        print(f"  Verifying MD5...")
        md5 = hashlib.md5()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        if md5.hexdigest() != checksum:
            archive_path.unlink()
            raise RuntimeError(f"MD5 mismatch for {name}. File deleted — re-run to retry.")

    # Extract
    print(f"  Extracting to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_type = info["type"]
    if archive_type == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(dest_dir)
    elif archive_type == "gz":
        out_path = dest_dir / info["filename"].replace(".gz", "")
        with gzip.open(archive_path, "rb") as gz_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out)

    # Remove archive to save disk space
    archive_path.unlink()
    print(f"  Done: {dest_dir}")
    return dest_dir


def verify_dataset(name: str) -> dict:
    """
    Check if a dataset is present and count its files.
    Returns a status dict.
    """
    if name not in DATASETS:
        return {"name": name, "status": "unknown_dataset"}

    info = DATASETS[name]
    dest_dir = DATA_DIR / info["extract_to"]

    if not dest_dir.exists():
        return {"name": name, "status": "missing", "path": str(dest_dir)}

    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    file_counts = {}
    total = 0
    for ext in audio_exts:
        count = len(list(dest_dir.rglob(f"*{ext}")))
        if count:
            file_counts[ext] = count
            total += count

    return {
        "name": name,
        "status": "present",
        "path": str(dest_dir),
        "audio_files": total,
        "by_extension": file_counts,
        "description": info["description"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download DreamStage training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download GTZAN (1.2 GB, fast, sufficient for genre training):
    python download.py --datasets gtzan

  Download FMA metadata (20 MB) then FMA small (7.2 GB):
    python download.py --datasets fma_metadata fma_small

  Download everything (fast datasets only, ~8 GB):
    python download.py --datasets gtzan fma_metadata fma_small

  List dataset status:
    python download.py --list
        """,
    )
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS) + ["all"],
                        default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download existing")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    if args.list:
        print(f"\nDataset directory: {DATA_DIR}\n")
        print(f"{'Name':<25} {'Size':>7}  {'Status':<12}  Description")
        print("-" * 80)
        for name, info in DATASETS.items():
            status = verify_dataset(name)
            s = status.get("status", "?")
            files = f"{status.get('audio_files', 0)} files" if s == "present" else ""
            print(f"{name:<25} {info['size_gb']:>6.1f}G  {s:<12}  {info['description'][:40]}")
        return

    names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    if not names:
        parser.print_help()
        return

    print(f"\nTarget directory: {DATA_DIR}")
    errors = []
    for name in names:
        try:
            download_dataset(name, force=args.force)
        except Exception as exc:
            print(f"\nERROR downloading {name}: {exc}")
            errors.append(name)

    if errors:
        print(f"\nFailed: {errors}")
        sys.exit(1)
    else:
        print("\nAll datasets downloaded successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
