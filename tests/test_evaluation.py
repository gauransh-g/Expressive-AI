"""
Unit tests for evaluation metrics (ROUGE, consistency, MOS).
"""

import pytest
from src.evaluation import Evaluator


def test_rouge_calculation():
    evaluator = Evaluator()
    ref = "The quick brown fox jumps over the lazy sleeping dog in the sunny park."
    summ = "A quick brown fox jumps over the lazy dog."

    scores = evaluator.calculate_rouge_scores(ref, summ)
    assert "rouge1_f" in scores
    assert "rouge2_f" in scores
    assert "rougeL_f" in scores
    assert scores["rouge1_f"] > 0.40


def test_consistency_check():
    evaluator = Evaluator()
    orig = "Dr. Robert Smith published research on quantum algorithms at MIT in Cambridge."
    summ = "Dr. Robert Smith studied quantum algorithms at MIT."

    result = evaluator.consistency_check(orig, summ)
    assert "consistency_score" in result
    assert "factual_consistency_score" in result
    assert "entity_retention_rate" in result
    assert result["entity_retention_rate"] >= 0.50
    assert result["factual_consistency_score"] >= 0.80


def test_repetition_metrics():
    evaluator = Evaluator()
    clean_text = "Transformer models generate high fidelity speech and concise summaries."
    rep_text = "The model is good. The model is good. The model is good."
    
    clean_res = evaluator.calculate_repetition_metrics(clean_text)
    rep_res = evaluator.calculate_repetition_metrics(rep_text)
    
    assert clean_res["repetition_cleanliness_score"] > rep_res["repetition_cleanliness_score"]


def test_compare_summaries():
    evaluator = Evaluator()
    orig = "CloudCorp Billing issued a $480 credit for Kubernetes cluster prj-prod-us-east-8842."
    base_summ = "A credit was issued."
    imp_summ = "CloudCorp issued a $480 credit for Kubernetes cluster prj-prod-us-east-8842."
    
    comp = evaluator.compare_summaries(orig, base_summ, imp_summ, target_words=15)
    assert "baseline" in comp
    assert "improved" in comp
    assert "deltas" in comp
    assert comp["improved"]["entity_retention"] >= comp["baseline"]["entity_retention"]


def test_mos_template_generation():
    evaluator = Evaluator()
    template = evaluator.generate_mos_template(
        audio_path="data/audio/generated_speeches/audio_conv_0001.wav",
        reference_text="Sample text",
        sample_id="conv_0001",
    )
    assert template["sample_id"] == "conv_0001"
    assert "intelligibility" in template["evaluation_criteria"]
    assert "naturalness" in template["evaluation_criteria"]
