"""
Unit tests for text summarization module.
"""

import pytest
from src.summarization import TextSummarizer


def test_summarizer_initialization():
    summarizer = TextSummarizer(model_name="facebook/bart-large-cnn", use_lazy_loading=True)
    assert summarizer.model_name == "facebook/bart-large-cnn"
    assert summarizer.max_length == 150
    assert summarizer.min_length == 30


def test_summarize_compression():
    summarizer = TextSummarizer(use_lazy_loading=True, load_weights=False)
    sample_text = (
        "Artificial intelligence and deep learning models have undergone exponential growth in capability over the past decade. "
        "Modern transformer architectures rely on self-attention mechanisms to process sequential natural language information in parallel. "
        "This allows models to capture intricate semantic relationships across long paragraphs and conversational dialogues. "
        "Text summarization represents one of the most impactful applications of natural language processing, condensing long documents into concise overviews. "
        "When paired with neural speech synthesis systems, summaries can be converted to high quality lifelike voice audio for multi-modal consumption."
    )
    summary = summarizer.summarize(sample_text)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary.split()) <= len(sample_text.split())


def test_batch_summarization():
    summarizer = TextSummarizer(use_lazy_loading=True, load_weights=False)
    texts = [
        "First document discusses cloud computing architectures and serverless scaling paradigms for enterprise operations.",
        "Second document covers quantum computing qubits and error correction methods across cryogenic physical platforms.",
    ]
    summaries = summarizer.batch_summarize(texts, batch_size=2)
    assert len(summaries) == 2
    assert all(len(s) > 0 for s in summaries)
