"""
Text Preprocessing & Normalization Module for Transformer Summarization and TTS Pipeline.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils import get_logger

logger = get_logger("Preprocessing")

# Comprehensive English contractions mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Compile contractions pattern
CONTRACTIONS_RE = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in CONTRACTIONS.keys()) + r")\b",
    flags=re.IGNORECASE,
)


class TextPreprocessor:
    """
    Normalizes raw text for summarization and prepares synthesized speech output.
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        min_tokens: int = 10,
        expand_contractions_flag: bool = True,
        remove_special_chars_flag: bool = True,
        normalize_unicode: bool = True,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.expand_contractions_flag = expand_contractions_flag
        self.remove_special_chars_flag = remove_special_chars_flag
        self.normalize_unicode = normalize_unicode

    def expand_contractions(self, text: str) -> str:
        """Expand short contractions into their full grammatical equivalents."""
        def replace(match):
            key = match.group(0).lower()
            replacement = CONTRACTIONS.get(key, match.group(0))
            if match.group(0)[0].isupper():
                return replacement.capitalize()
            return replacement

        return CONTRACTIONS_RE.sub(replace, text)

    def clean_unicode(self, text: str) -> str:
        """Normalize Unicode characters (smart quotes, accents, non-breaking spaces)."""
        # Normalize NFKC
        text = unicodedata.normalize("NFKC", text)
        # Replace non-standard quotes and hyphens
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        return text

    def clean_special_characters(self, text: str) -> str:
        """Remove URLs, HTML tags, strange control characters while preserving sentence punctuation, numbers, currencies, and technical symbols."""
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Strip URLs but keep domain references if relevant
        text = re.sub(r"http[s]?://\S+", "", text)
        # Strip Markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Strip Markdown bold/italics formatting
        text = re.sub(r"[*_~`#]", "", text)
        # Retain alphanumeric, spaces, standard sentence punctuation, currencies ($ € £ ¥), %, +, =, /, @, -, :, ;
        text = re.sub(r"[^\w\s.,!?'\":;()$%&€£¥/+=@\-]", " ", text)
        return text

    def extract_salient_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts salient numbers, currency values, dates, percentages, technical identifiers, and proper nouns.
        """
        if not text:
            return {"numbers": [], "currencies": [], "entities": [], "technical_terms": []}

        # 1. Currencies & financial amounts
        currencies = re.findall(r"[\$€£¥]\s?\d+(?:[\.,]\d+)?(?:\s?(?:million|billion|trillion|k|M|B))?", text, re.IGNORECASE)
        
        # 2. Percentages & multipliers
        percentages = re.findall(r"\b\d+(?:\.\d+)?(?:%|x|X|\s?percent)\b", text, re.IGNORECASE)
        
        # 3. Numeric values with units or standalone numbers
        numbers = re.findall(r"\b\d+(?:[\.,]\d+)?(?:\s?(?:nodes|tokens|pods|instances|users|ms|seconds|minutes|hours|days|GB|MB|TB|GHz|MHz|nm|Hz))?\b", text, re.IGNORECASE)

        # 4. Technical IDs and codes (e.g., prj-prod-us-east-8842, User_921, v4.12.1, O(N^2))
        tech_ids = re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b|\bv?\d+\.\d+(?:\.\d+)?\b|\b[A-Z]{2,}(?:LM|TTS|API|SDK|GPU|CPU|RAM|AI|ML|NLP|MOS|ROUGE|UI)\b", text)

        # 5. Capitalized Proper Nouns / Names (e.g. Sarah M., CloudCorp, Kubernetes)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z\.]+)?\b", text)

        # Filter out trivial common sentence starters
        common_starters = {"The", "This", "That", "These", "Those", "There", "Here", "What", "When", "Where", "Why", "How", "If", "In", "On", "At", "For", "With", "As", "By", "To", "From", "Hello", "Thank", "Yes", "No", "Please", "Customer", "Agent"}
        filtered_proper_nouns = [p for p in proper_nouns if p not in common_starters and len(p) > 2]

        return {
            "currencies": list(dict.fromkeys(currencies)),
            "percentages": list(dict.fromkeys(percentages)),
            "numbers": list(dict.fromkeys(numbers)),
            "technical_terms": list(dict.fromkeys(tech_ids)),
            "entities": list(dict.fromkeys(filtered_proper_nouns)),
        }

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces, newlines, and tabs into clean single spaces."""
        text = re.sub(r"\r\n|\r|\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Split text into coherent sentences using punctuation heuristics."""
        sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
        sentences = [s.strip() for s in sentence_regex.split(text) if s.strip()]
        return sentences

    def is_english(self, text: str) -> bool:
        """Lightweight heuristic to check if the input is primarily Latin/English."""
        if not text:
            return False
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return (ascii_chars / len(text)) > 0.85

    def validate_and_truncate(self, text: str, max_words: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Validates text length and truncates if necessary to fit within token boundaries.
        """
        limit = max_words or self.max_tokens
        words = text.split()
        original_word_count = len(words)

        if original_word_count > limit:
            truncated_words = words[:limit]
            cleaned_truncated = " ".join(truncated_words)
            is_truncated = True
        else:
            cleaned_truncated = text
            is_truncated = False

        stats = {
            "original_word_count": original_word_count,
            "final_word_count": len(cleaned_truncated.split()),
            "is_truncated": is_truncated,
            "char_count": len(cleaned_truncated),
        }
        return cleaned_truncated, stats

    def preprocess(self, text: str) -> str:
        """
        Complete end-to-end preprocessing pipeline for an input string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.normalize_unicode:
            text = self.clean_unicode(text)

        if self.expand_contractions_flag:
            text = self.expand_contractions(text)

        if self.remove_special_chars_flag:
            text = self.clean_special_characters(text)

        text = self.normalize_whitespace(text)

        # Enforce max word limit
        text, _ = self.validate_and_truncate(text, self.max_tokens)
        return text


# Global preprocessor instance for fast standalone calls
_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Standard top-level functional interface for text preprocessing.
    """
    return _DEFAULT_PREPROCESSOR.preprocess(text)
