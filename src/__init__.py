"""
End-to-End Transformer Text Summarization + Neural TTS Pipeline.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

__version__ = "1.0.0"

from src.config import AppConfig, load_config
from src.preprocessing import TextPreprocessor, preprocess_text
from src.summarization import TextSummarizer
from src.tts_pipeline import TTSPipeline
from src.evaluation import Evaluator

__all__ = [
    "AppConfig",
    "load_config",
    "TextPreprocessor",
    "preprocess_text",
    "TextSummarizer",
    "TTSPipeline",
    "Evaluator",
]
