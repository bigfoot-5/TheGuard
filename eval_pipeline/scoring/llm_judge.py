import json
import os
import sys

# Dynamically import the centralized LLM runner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_runner import generate_response

# ==========================================
# HELPER: ROBUST JSON PARSER
# ==========================================
def _extract_json(raw_text: str) -> dict:
    """
    Bulletproof JSON extractor. Hunts for the outermost brackets, 
    completely ignoring any conversational filler from the LLM.
    """
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = raw_text[start_idx:end_idx+1]
            return json.loads(json_str)
        print("  -> ⚠️ Warning: No JSON brackets found in Judge response.")
        return {}
    except Exception as e:
        print(f"  -> ⚠️ Warning: Failed to parse Judge JSON: {e}")
        return {}

# ==========================================
# METRIC 1: FACTUAL GROUNDING (NLI)
# ==========================================
def score_factual_grounding(source_data: dict, generated_narrative: str, judge_model: str = "gpt-4o") -> tuple[float, float]:
    """
    Evaluates if the narrative hallucinates facts not present in the source data.
    Returns (score, cost)
    """
    system_prompt = """
    You are an expert underwriter evaluation AI. Your job is Natural Language Inference (NLI).
    You will be given JSON Source Data and a Generated Narrative.
    1. Break the narrative down into distinct, atomic factual claims.
    2. Compare each claim to the Source Data.
    3. Classify each as:
       - 'Entailment' (supported by source)
       - 'Contradiction' (conflicts with source)
       - 'Neutral' (not mentioned in source / hallucinated)

    Respond ONLY with valid JSON. Example format:
    {
      "claims": [
        {
          "atomic_claim": "The merchant has been operating for 5 years.",
          "classification": "Entailment",
          "reasoning": "Matches business_vintage_years perfectly."
        }
      ]
    }
    """
    
    user_prompt = f"Source Data: {json.dumps(source_data)}\nGenerated Narrative: {generated_narrative}"

    # Send through the Centralized Gateway!
    raw_response, cost = generate_response(system_prompt, user_prompt, model=judge_model, temperature=0.0)
    
    # Parse output using our universal extractor
    parsed_data = _extract_json(raw_response)
    evaluations = parsed_data.get("claims", [])
    
    if not evaluations:
        return 0.0, cost
        
    total_claims = len(evaluations)
    entailed_claims = sum(1 for claim in evaluations if claim.get("classification") == "Entailment")
    
    final_score = round(entailed_claims / total_claims, 4) if total_claims > 0 else 0.0
    return final_score, cost


# ==========================================
# METRIC 2: PERSUASIVENESS QUALITY
# ==========================================
def score_persuasiveness(generated_copy: str, judge_model: str = "gpt-4o") -> tuple[float, float]:
    """
    Evaluates marketing copy on Clarity, Urgency, and CTA Strength.
    Returns (normalized_score, cost)
    """
    system_prompt = """
    You are an expert Chief Marketing Officer evaluating promotional copy.
    Rate the provided copy from 1 to 10 on three axes:
    1. Clarity: Is the discount/offer immediately obvious?
    2. Urgency: Does it compel the user to act quickly without sounding spammy?
    3. CTA Strength: Is the Call-To-Action clear and actionable?

    Respond ONLY with valid JSON. Example format:
    {
      "clarity": 8,
      "urgency": 9,
      "cta_strength": 7,
      "feedback": "Great urgency, but CTA could be slightly more direct."
    }
    """

    user_prompt = f"Evaluate this copy: {generated_copy}"

    # Send through the Centralized Gateway!
    raw_response, cost = generate_response(system_prompt, user_prompt, model=judge_model, temperature=0.2)
    
    # Parse output
    result = _extract_json(raw_response)
    
    total_score = result.get("clarity", 0) + result.get("urgency", 0) + result.get("cta_strength", 0)
    normalized_score = round(total_score / 30.0, 4)
    
    return normalized_score, cost