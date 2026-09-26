"""
Unit tests for text preprocessing and normalization.
"""

import pytest
from src.preprocessing import TextPreprocessor, preprocess_text


def test_contractions_expansion():
    preprocessor = TextPreprocessor()
    raw = "I can't believe it's not working, they're going to fix it."
    cleaned = preprocessor.expand_contractions(raw)
    assert "cannot" in cleaned
    assert "it is" in cleaned
    assert "they are" in cleaned


def test_special_characters_removal():
    preprocessor = TextPreprocessor()
    raw = "Here is a link <p>http://example.com/api</p> with **bold** markup and $100 price tag."
    cleaned = preprocessor.clean_special_characters(raw)
    assert "<p>" not in cleaned
    assert "http://" not in cleaned
    assert "**" not in cleaned
    assert "$100" in cleaned


def test_whitespace_normalization():
    preprocessor = TextPreprocessor()
    raw = "   This   has \n\n  multiple \t lines   and spaces.   "
    cleaned = preprocessor.normalize_whitespace(raw)
    assert cleaned == "This has multiple lines and spaces."


def test_sentence_segmentation():
    preprocessor = TextPreprocessor()
    raw = "First sentence here. Second sentence starts now! What about the third one? Yes, indeed."
    sentences = preprocessor.segment_sentences(raw)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence here."


def test_end_to_end_preprocess_text():
    raw = "Hello! We're testing the <mark>pipeline</mark> with http://test.com at $500 value."
    cleaned = preprocess_text(raw)
    assert "We are" in cleaned
    assert "<mark>" not in cleaned
    assert "http" not in cleaned
    assert "$500" in cleaned
