import json
from scoring.format_compliance import score_format_compliance
from scoring.semantic_similarity import score_semantic_similarity
from scoring.llm_judge import score_factual_grounding, score_persuasiveness

# Standardized Wrappers: Every metric takes (case_dict, generated_output)
def wrap_format_compliance(case, output):
    return score_format_compliance(output, case['expected_max_length'])

def wrap_semantic_similarity(case, output):
    return score_semantic_similarity(case['raw_deal_extraction'], output)

def wrap_factual_grounding(case, output):
    return score_factual_grounding(case, output)

def wrap_intent_accuracy(case, output):
    try:
        output_json = json.loads(output.strip('` \n').replace('json\n', ''))
        return 1.0 if output_json.get("intent") == case['expected_intent'] else 0.0
    except:
        return 0.0

# The Dictionary that links the YAML string to the Python function
REGISTRY = {
    "format_compliance": wrap_format_compliance,
    "semantic_similarity": wrap_semantic_similarity,
    "factual_grounding": wrap_factual_grounding,
    "intent_accuracy": wrap_intent_accuracy
}