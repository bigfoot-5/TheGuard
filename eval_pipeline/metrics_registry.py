import json
from scoring.format_compliance import score_format_compliance, score_email_leak
from scoring.semantic_similarity import score_semantic_similarity
from scoring.llm_judge import score_factual_grounding, score_persuasiveness
from scoring.confidence_calibration import score_confidence_calibration, score_intent_accuracy, score_edge_case_handling

def wrap_format_compliance(case, output, **kwargs):
    return score_format_compliance(case, output)

def wrap_semantic_similarity(case, output, **kwargs):
    emb_model = kwargs.get("embedding_model", "text-embedding-3-small")
    return score_semantic_similarity(case['raw_deal_extraction'], output, embedding_model=emb_model)

def wrap_factual_grounding(case, output, **kwargs):
    judge = kwargs.get("judge_model", "gpt-4o-mini")
    return score_factual_grounding(case, output, judge_model=judge)

def wrap_persuasiveness(case, output, **kwargs):
    judge = kwargs.get("judge_model", "gpt-4o-mini")
    return score_persuasiveness(output, judge_model=judge)

def wrap_intent_accuracy(case, output, **kwargs):
    return score_intent_accuracy(case, output)

def wrap_confidence_calibration(case, output, **kwargs):
    return score_confidence_calibration(case, output)

def wrap_edge_case_handling(case, output, **kwargs):
    return score_edge_case_handling(case, output)

def wrap_email_leak(case, output, **kwargs):
    return score_email_leak(output)

# Maps the metric names from the config to the functions above
REGISTRY = {
    "format_compliance": wrap_format_compliance,
    "semantic_similarity": wrap_semantic_similarity,
    "factual_grounding": wrap_factual_grounding,
    "persuasiveness": wrap_persuasiveness,
    "intent_accuracy": wrap_intent_accuracy,
    "confidence_calibration": wrap_confidence_calibration,
    "edge_case_handling": wrap_edge_case_handling,
    "email_leak": wrap_email_leak
}