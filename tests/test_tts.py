"""
Unit tests for neural TTS pipeline and audio generation.
"""

import os
import tempfile
import pytest
from src.tts_pipeline import TTSPipeline
from src.utils import get_audio_metadata


def test_tts_initialization():
    tts = TTSPipeline(sample_rate=22050)
    assert tts.sample_rate == 22050
    assert tts.engine_type in ["auto", "coqui", "pyttsx3", "gtts", "edge_tts_neural", "synth_fallback"]


def test_audio_synthesis_and_metadata():
    tts = TTSPipeline(sample_rate=22050, engine="synth_fallback")
    text = "This is a verification test for the neural text to speech synthesis pipeline."

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_audio.wav")
        meta = tts.synthesize(text, wav_path)

        assert os.path.exists(wav_path)
        assert meta["exists"] is True
        assert meta["sample_rate"] == 22050
        assert meta["duration_seconds"] > 0.5
        assert meta["rms_energy"] > 0.0


def test_batch_synthesis():
    tts = TTSPipeline(sample_rate=22050, engine="synth_fallback")
    items = [
        {"id": "test_01", "summary": "First summary audio item."},
        {"id": "test_02", "summary": "Second summary audio item."},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results = tts.batch_synthesize(items, output_dir=tmpdir)
        assert len(results) == 2
        for res in results:
            assert os.path.exists(res["file_path"])
            assert res["duration_seconds"] > 0.0
