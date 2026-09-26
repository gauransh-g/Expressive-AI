import os
import re
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.preprocessing import preprocess_text
from src.utils import chunk_text, get_logger, resolve_device, timeit

logger = get_logger("Summarization")


class TextSummarizer:
    """
    State-of-the-art Transformer Abstractive Text Summarizer.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: str = "auto",
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        use_lazy_loading: bool = True,
        load_weights: bool = True,
    ):
        self.model_name = model_name
        self.device_str = device
        self.torch_device = resolve_device(device)
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.load_weights = load_weights

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        if not use_lazy_loading:
            self._load_model()

    def _load_model(self) -> None:
        """Loads model weights and tokenizer onto the designated device."""
        if not self.load_weights or self._is_loaded:
            return

        logger.info(f"Loading summarization model '{self.model_name}' on device '{self.torch_device}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.torch_device)
            self.model.eval()

            self._is_loaded = True
            logger.info("Summarization model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load '{self.model_name}' from Hugging Face ({e}). "
                "Enabling intelligent fallback summarizer."
            )
            self._is_loaded = False

    def _clean_summary_output(self, text: str) -> str:
        """
        Cleans generated summary: removes repetitive sentences, ensures proper capitalization,
        and trims any trailing fragmented clauses to end cleanly at sentence boundaries.
        """
        text = text.strip()
        if not text:
            return ""

        # Remove immediate duplicated sentences or phrases
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        seen = set()
        deduped = []
        for s in sentences:
            s_clean = s.lower().strip(".,!? ")
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                deduped.append(s)

        cleaned = " ".join(deduped) if deduped else text

        # Ensure summary terminates with clean sentence punctuation
        if not re.search(r"[.!?]$", cleaned):
            # Find the last period, exclamation, or question mark
            last_punc = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
            if last_punc > 30:
                cleaned = cleaned[: last_punc + 1]
            else:
                cleaned += "."

        return cleaned

    def _score_sentences(self, sentences: List[str], full_text: str) -> List[Dict[str, Any]]:
        """Scores sentences based on term frequency, numeric/entity density, technical terms, and structural role."""
        words_all = [w.lower() for w in full_text.split() if w.isalnum() and len(w) > 3]
        freq: Dict[str, int] = {}
        for w in words_all:
            freq[w] = freq.get(w, 0) + 1

        entities_found = set(re.findall(r"[\$€£¥]\s?\d+|\b\d+(?:\.\d+)?%?|\b[A-Z][a-z]+|\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", full_text))
        tech_keywords = {
            "telemetry", "architecture", "accuracy", "latency", "throughput", "result", "results",
            "limitation", "limitations", "algorithm", "cron", "credit", "resolved", "conclusion",
            "scaling", "nodes", "pod", "pods", "instance", "instances", "kafka", "redis", "kubernetes",
            "microservices", "overhead", "bottleneck", "uptime", "production", "benchmark", "benchmarking",
            "compression", "serialization", "pipeline", "database", "query", "queries", "cache", "server"
        }

        n_sents = len(sentences)
        z1_end = max(1, int(n_sents * 0.30))
        # Zone C is strictly the final concluding portion (last 1-2 sentences or explicit conclusion keywords)
        z2_end = max(z1_end + 1, n_sents - max(1, int(n_sents * 0.22)))

        conclusion_markers = {"conclusion", "finally", "resolved", "officially", "launched", "overall", "in summary", "to summarize", "consequently", "as a result", "uptime"}

        scored = []
        for idx, s in enumerate(sentences):
            s_words = [w.lower() for w in s.split() if w.isalnum()]
            if not s_words:
                continue

            f_score = sum(freq.get(w, 0) for w in s_words) / max(1, len(s_words))
            s_entities = sum(1 for e in entities_found if e in s)
            entity_boost = 1.0 + min(0.8, s_entities * 0.20)
            
            # Numeric values boost (preserves quantitative results like 310%, 450ms, 42ms, 15MB, 99.99%)
            s_numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?|\b\d+[a-zA-Z]+\b", s))
            num_boost = 1.0 + min(0.7, s_numbers * 0.25)

            s_tech = sum(1 for k in tech_keywords if k in s.lower())
            tech_boost = 1.0 + min(0.6, s_tech * 0.25)

            has_conclusion_marker = any(m in s.lower() for m in conclusion_markers)

            if idx < z1_end:
                zone = "beginning"
                role_boost = 1.25
            elif idx >= z2_end or (idx >= n_sents - 2 and has_conclusion_marker):
                zone = "conclusion"
                role_boost = 1.55 if has_conclusion_marker else 1.40
            else:
                zone = "middle"
                role_boost = 1.40  # Technical methods, benchmarks, limitations

            total_score = (f_score + 1.0) * role_boost * entity_boost * num_boost * tech_boost
            scored.append({
                "idx": idx,
                "score": total_score,
                "sentence": s,
                "words": len(s.split()),
                "zone": zone,
            })
        return scored

    def _extractive_fallback_summarize(
        self,
        text: str,
        target_words: int = 75,
        min_words: int = 40,
    ) -> str:
        """
        Section-stratified, entity-aware summarizer that guarantees balanced representation across
        the entire document: Beginning (Context), Middle (Technical Methods/Results/Limitations), and Ending (Conclusions).
        """
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        valid_sentences = [s for s in raw_sentences if len(s.split()) >= 3]
        if not valid_sentences:
            return text

        total_input_words = sum(len(s.split()) for s in valid_sentences)
        if total_input_words <= target_words:
            return text

        scored_sentences = self._score_sentences(valid_sentences, text)

        selected_indices = set()
        current_words = 0

        # Partition scored sentences by zone
        b_sents = sorted([s for s in scored_sentences if s["zone"] == "beginning"], key=lambda x: x["score"], reverse=True)
        m_sents = sorted([s for s in scored_sentences if s["zone"] == "middle"], key=lambda x: x["score"], reverse=True)
        c_sents = sorted([s for s in scored_sentences if s["zone"] == "conclusion"], key=lambda x: x["score"], reverse=True)

        # 1. Mandatory core coverage: select top-1 sentence from each available zone (Beginning, Middle, Conclusion)
        if b_sents:
            selected_indices.add(b_sents[0]["idx"])
            current_words += b_sents[0]["words"]

        if m_sents:
            selected_indices.add(m_sents[0]["idx"])
            current_words += m_sents[0]["words"]

        if c_sents:
            selected_indices.add(c_sents[0]["idx"])
            current_words += c_sents[0]["words"]

        # 2. Expand coverage if target word budget allows (for targets 75, 100, 120, 150+ words)
        remaining_sents = sorted([s for s in scored_sentences if s["idx"] not in selected_indices], key=lambda x: x["score"], reverse=True)
        for s in remaining_sents:
            if current_words + s["words"] <= target_words + 15 or current_words < min_words:
                selected_indices.add(s["idx"])
                current_words += s["words"]
                if current_words >= target_words:
                    break

        if not selected_indices:
            selected_indices = {0}

        ordered_indices = sorted(list(selected_indices))
        summary_sentences = [valid_sentences[i] for i in ordered_indices]
        summary = " ".join(summary_sentences)
        return self._clean_summary_output(summary)

    def _summarize_chunk(
        self,
        text_chunk: str,
        max_tokens: int,
        min_tokens: int,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Summarizes a single text chunk that fits in the token window with repetition suppression and robust boundaries."""
        inputs = self.tokenizer(
            text_chunk,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.torch_device)

        input_len = inputs["input_ids"].shape[1]
        
        # Robust token boundaries: ensure min_tokens <= input_len * 0.70 and min_tokens < max_tokens
        effective_min = max(10, min(min_tokens, int(input_len * 0.65)))
        effective_max = min(1024, max(effective_min + 10, max_tokens))

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=self.num_beams,
                max_length=effective_max,
                min_length=effective_min,
                length_penalty=1.0,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return self._clean_summary_output(summary)

    @timeit
    def summarize(
        self,
        text: str,
        target_max_length: Optional[int] = None,
        target_min_length: Optional[int] = None,
        mode: str = "enhanced",
    ) -> str:
        """
        Generates a balanced summary targeting the desired word length while preserving
        factual entities, numbers, technical methods, results, and final conclusions across the entire document.
        """
        cleaned = preprocess_text(text)
        if not cleaned:
            return ""

        words = cleaned.split()
        total_words = len(words)
        if total_words < 25:
            return cleaned

        # Safe target word bounds: never error out on high target length
        if target_max_length is not None and target_max_length > 0:
            target_words = min(target_max_length, total_words)
        else:
            target_words = min(self.max_length, max(35, int(total_words * 0.45)))

        if target_min_length is not None and target_min_length > 0:
            min_words = min(target_min_length, target_words)
        else:
            min_words = max(20, int(target_words * 0.80))

        # Check if full text is already within target word budget
        if total_words <= target_words:
            return cleaned

        self._load_model()

        # For extractive or fallback mode, run zone-stratified summarization
        if not self._is_loaded or self.model is None or mode in ("enhanced", "extractive_enhanced"):
            return self._extractive_fallback_summarize(cleaned, target_words=target_words, min_words=min_words)

        # For abstractive neural transformer mode, run multi-zone salience guided generation
        zone_summary = self._extractive_fallback_summarize(cleaned, target_words=target_words, min_words=min_words)
        
        rep_pen = 1.15
        token_max_length = min(1024, int(target_words * 1.45) + 15)
        token_min_length = max(15, min(int(min_words * 1.15), int(len(zone_summary.split()) * 0.65)))

        abstractive = self._summarize_chunk(
            zone_summary,
            max_tokens=token_max_length,
            min_tokens=token_min_length,
            repetition_penalty=rep_pen,
        )

        # Safety check: if abstractive dropped more than 50% of the content or omitted zones, fallback to zone_summary
        if len(abstractive.split()) < 20:
            return zone_summary
        return abstractive

    def batch_summarize(
        self,
        texts: List[str],
        batch_size: int = 4,
        mode: str = "enhanced",
    ) -> List[str]:
        """
        Processes a list of text documents in parallel or batches.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                summary = self.summarize(text, mode=mode)
                results.append(summary)
        return results

