import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
client = OpenAI() # Automatically picks up OPENAI_API_KEY from environment

# ==========================================
# PYDANTIC SCHEMAS FOR STRUCTURED LLM OUTPUT
# ==========================================

# Schemas for Factual Grounding (NLI)
class ClaimEvaluation(BaseModel):
    atomic_claim: str = Field(description="A single, independent factual claim extracted from the narrative.")
    classification: str = Field(description="Must be one of: Entailment, Contradiction, Neutral")
    reasoning: str = Field(description="Brief explanation of why this classification was chosen based on source data.")

class GroundingResult(BaseModel):
    claims: list[ClaimEvaluation]

# Schema for Persuasiveness Quality
class QualityScores(BaseModel):
    clarity: int = Field(description="Score from 1 to 10")
    urgency: int = Field(description="Score from 1 to 10")
    cta_strength: int = Field(description="Score from 1 to 10")
    feedback: str = Field(description="One sentence of constructive feedback.")

# ==========================================
# METRIC 1: FACTUAL GROUNDING (NLI)
# ==========================================
def score_factual_grounding(source_data: dict, generated_narrative: str) -> float:
    """
    Evaluates if the narrative hallucinates facts not present in the source data.
    Score = N_entailed / N_total_claims
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
    """
    
    user_prompt = f"""
    Source Data: {json.dumps(source_data)}
    Generated Narrative: {generated_narrative}
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=GroundingResult,
        temperature=0.0 # Strict, deterministic evaluation
    )
    
    evaluations = response.choices[0].message.parsed.claims
    
    if not evaluations:
        return 0.0
        
    # Calculate Math: Score = Entailed / Total
    total_claims = len(evaluations)
    entailed_claims = sum(1 for claim in evaluations if claim.classification == "Entailment")
    
    score = entailed_claims / total_claims
    return round(score, 4)

# ==========================================
# METRIC 2: PERSUASIVENESS QUALITY
# ==========================================
def score_persuasiveness(generated_copy: str) -> float:
    """
    Evaluates marketing copy on Clarity, Urgency, and CTA Strength (1-10 each).
    Normalizes the sum to a 0.0 - 1.0 scale.
    """
    system_prompt = """
    You are an expert Chief Marketing Officer evaluating promotional copy.
    Rate the provided copy from 1 to 10 on three axes:
    1. Clarity: Is the discount/offer immediately obvious?
    2. Urgency: Does it compel the user to act quickly without sounding spammy?
    3. CTA Strength: Is the Call-To-Action clear and actionable?
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Evaluate this copy: {generated_copy}"}
        ],
        response_format=QualityScores,
        temperature=0.2
    )
    
    result = response.choices[0].message.parsed
    
    # Calculate Math: Max possible score is 30 (10 + 10 + 10)
    total_score = result.clarity + result.urgency + result.cta_strength
    normalized_score = total_score / 30.0
    
    return round(normalized_score, 4)

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Factual Grounding (NLI)...")
    source = {"business_vintage_years": 2.5, "yoy_gmv_growth_percentage": 14.5, "historical_default_rate_percentage": 0.0}
    
    # Good Narrative: Sticks to the facts
    good_narrative = "The merchant has been operating for 2.5 years with no historical defaults and a 14.5% GMV growth."
    print(f"Good Grounding Score: {score_factual_grounding(source, good_narrative)}") # Expected: 1.0
    
    # Hallucinated Narrative: Makes up a 38% growth rate
    bad_narrative = "The merchant has been operating for 2.5 years and shows an incredible 38% GMV growth."
    print(f"Bad Grounding Score: {score_factual_grounding(source, bad_narrative)}") # Expected: ~0.5
    
    print("\n🧪 Testing Persuasiveness Quality...")
    boring_copy = "We have laptops for sale. You can buy them if you want. Use code MAC50."
    print(f"Boring Copy Score: {score_persuasiveness(boring_copy)}") # Expected: Low score (< 0.5)
    
    fire_copy = "FLASH SALE! 🚨 Get 50% off MacBooks today only! Stock is strictly limited. Use code MAC50 at checkout right now!"
    print(f"Fire Copy Score: {score_persuasiveness(fire_copy)}") # Expected: High score (> 0.85)