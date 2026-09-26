"""
Utility helpers for device selection, logging, I/O, audio processing, and metrics.
"""

import json
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def get_logger(name: str = "NeuralTTSPipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a clean, structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("Utils")


def resolve_device(preferred: str = "auto") -> torch.device:
    """
    Resolves the torch computing device ('cuda', 'mps', 'cpu', or 'auto').
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def timeit(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Atomically write data as formatted JSON."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON to: {file_path}")


def load_json(file_path: str) -> Any:
    """Load and parse JSON file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_compression_ratio(original_text: str, summary_text: str) -> float:
    """
    Calculates the token/word length compression ratio (summary_len / original_len).
    """
    orig_words = len(original_text.split())
    if orig_words == 0:
        return 0.0
    summ_words = len(summary_text.split())
    return round(summ_words / orig_words, 4)


def get_audio_metadata(audio_path: str) -> Dict[str, Any]:
    """
    Extracts duration, sample rate, channels, RMS energy, and silence ratio of an audio file.
    """
    import soundfile as sf

    if not os.path.exists(audio_path):
        return {"exists": False, "error": "File not found"}

    try:
        data, samplerate = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)  # convert to mono for metrics

        duration = len(data) / float(samplerate)
        rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
        peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0

        # Estimate silence ratio (below -40dB relative to peak)
        threshold = max(peak * 0.01, 1e-4)
        silence_samples = np.sum(np.abs(data) < threshold)
        silence_ratio = float(silence_samples / len(data)) if len(data) > 0 else 0.0

        return {
            "exists": True,
            "file_path": audio_path,
            "sample_rate": samplerate,
            "duration_seconds": round(duration, 3),
            "num_samples": len(data),
            "rms_energy": round(rms, 4),
            "peak_amplitude": round(peak, 4),
            "silence_ratio": round(silence_ratio, 4),
            "is_clipped": peak >= 0.999,
        }
    except Exception as e:
        logger.warning(f"Error reading audio metadata from {audio_path}: {e}")
        return {"exists": True, "error": str(e)}


def chunk_text(text: str, max_words_per_chunk: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits long text into overlapping chunks if it exceeds the maximum token length.
    """
    words = text.split()
    if len(words) <= max_words_per_chunk:
        return [text]

    chunks = []
    start = 0
    step = max_words_per_chunk - overlap
    while start < len(words):
        chunk = " ".join(words[start : start + max_words_per_chunk])
        chunks.append(chunk)
        start += step
        if start + overlap >= len(words):
            break
    return chunks
