"""
Configuration management module for the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os
import yaml


@dataclass
class SummarizationConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 150
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    device: str = "auto"
    batch_size: int = 4
    chunk_overlap: int = 50
    max_chunk_tokens: int = 1024


@dataclass
class TTSConfig:
    model_name: str = "tts_models/en/ljspeech/glow-tts"
    sample_rate: int = 22050
    device: str = "auto"
    speaker: str = "default"
    speed: float = 1.0
    pitch_shift: float = 0.0
    normalize_audio: bool = True
    trim_silence: bool = True
    engine_fallback_order: List[str] = field(
        default_factory=lambda: ["coqui", "hf_speecht5", "pyttsx3", "gtts"]
    )


@dataclass
class PreprocessingConfig:
    max_input_length: int = 1024
    min_input_length: int = 20
    language: str = "en"
    remove_special_chars: bool = True
    expand_contractions: bool = True
    normalize_whitespace: bool = True
    sentence_segmentation: bool = True


@dataclass
class EvaluationConfig:
    rouge_metrics: List[str] = field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
    mos_scale: int = 5
    consistency_threshold: float = 0.70
    semantic_similarity_model: str = "all-MiniLM-L6-v2"
    check_entity_preservation: bool = True
    evaluate_audio_snr: bool = True


@dataclass
class DataConfig:
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    audio_output_dir: str = "data/audio/generated_speeches"
    results_dir: str = "results"
    raw_samples_file: str = "data/raw/sample_conversations.json"
    summarized_output_file: str = "data/processed/summarized_outputs.json"
    evaluation_metrics_file: str = "results/evaluation_metrics.json"
    mos_scores_file: str = "results/mos_scores.csv"
    num_workers: int = 4
    batch_size: int = 8


@dataclass
class AppConfig:
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def ensure_directories(self):
        """Ensure all required input, output and results directories exist."""
        for path_str in [
            self.data.input_dir,
            self.data.output_dir,
            self.data.audio_output_dir,
            self.data.results_dir,
            os.path.join(self.data.results_dir, "sample_outputs"),
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML file or return defaults."""
    if config_path is None or not os.path.exists(config_path):
        default_yaml = Path(__file__).resolve().parent.parent / "config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        sum_cfg = SummarizationConfig(**data.get("summarization", {}))
        tts_cfg = TTSConfig(**data.get("tts", {}))
        prep_cfg = PreprocessingConfig(**data.get("preprocessing", {}))
        eval_cfg = EvaluationConfig(**data.get("evaluation", {}))
        data_cfg = DataConfig(**data.get("data", {}))

        cfg = AppConfig(
            summarization=sum_cfg,
            tts=tts_cfg,
            preprocessing=prep_cfg,
            evaluation=eval_cfg,
            data=data_cfg,
        )
    else:
        cfg = AppConfig()

    cfg.ensure_directories()
    return cfg
