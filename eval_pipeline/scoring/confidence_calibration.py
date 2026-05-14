import json

def _parse_llm_json(raw_text: str) -> dict:
    """
    Bulletproof JSON extractor. Hunts for the outermost brackets, 
    completely ignoring Claude's conversational filler.
    """
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = raw_text[start_idx:end_idx+1]
            return json.loads(json_str)
            
        print("  -> ⚠️ Warning: No JSON brackets found in the response.")
        return {}
    except Exception as e:
        print(f"  -> ⚠️ Warning: Failed to parse LLM JSON output: {e}")
        return {}

# ==========================================
# METRIC 1: INTENT ACCURACY (BINARY)
# ==========================================
def score_intent_accuracy(case: dict, generated_text: str) -> float:
    """
    Binary metric: Returns 1.0 if the predicted intent matches the expected intent exactly.
    """
    expected_intent = case.get("expected_intent", "").strip().lower()
    
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    
    # Strict Pass/Fail
    if predicted_intent == expected_intent and expected_intent != "":
        return 1.0
    return 0.0

# ==========================================
# METRIC 2: CONFIDENCE CALIBRATION (CONTINUOUS)
# ==========================================
def score_confidence_calibration(case: dict, generated_text: str) -> float:
    """
    Continuous metric: Calculates Mean Absolute Calibration Error (MACE) for a single case.
    Inverts the score so 1.0 is perfect calibration, and 0.0 is a confident hallucination.
    """
    expected_intent = case.get("expected_intent", "").strip().lower()
    
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    
    # Default to 0.0 if the LLM forgot to include a confidence score
    confidence = float(output_data.get("confidence_score", 0.0))
    
    # 1. Did the model get it right?
    is_correct = (predicted_intent == expected_intent)
    
    # 2. What SHOULD its confidence have been?
    # If it is correct, it should be 100% (1.0) confident.
    # If it is wrong, it should be 0% (0.0) confident.
    ideal_confidence = 1.0 if is_correct else 0.0
    
    # 3. Calculate the Error (Distance from reality)
    error = abs(ideal_confidence - confidence)
    
    # 4. Invert so Higher is Better (to work with your Dashboard trendlines)
    # Perfect calibration = 1.0. Terrible overconfidence = 0.0.
    return 1.0 - error

def score_edge_case_handling(case: dict, generated_text: str) -> float:
    """
    Scores how safely the model handles ambiguous carts.
    To pass an edge case, the model MUST either:
    1. Output 'No Insurance Applicable' OR
    2. Output a confidence score below 0.60 (admitting it is unsure).
    """
    is_edge_case = case.get("is_ambiguous_edge_case", False)
    
    # If it's a standard, easy case, this specific metric automatically passes.
    if not is_edge_case:
        return 1.0 
        
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    confidence = float(output_data.get("confidence_score", 1.0))
    
    # Did the model handle the trap safely?
    if predicted_intent == "no insurance applicable":
        return 1.0 # Safest fallback!
    elif confidence <= 0.60:
        return 1.0 # It guessed, but correctly flagged its own uncertainty.
    else:
        return 0.0

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    test_case = {"expected_intent": "Travel Cancellation"}
    
    # Scenario 1: Correct and highly confident
    good_output = '{"intent": "Travel Cancellation", "confidence_score": 0.95}'
    print(f"Accuracy: {score_intent_accuracy(test_case, good_output)}")     # Expected: 1.0
    print(f"Calibration: {score_confidence_calibration(test_case, good_output)}\n") # Expected: ~0.95
    
    # Scenario 2: Wrong, but the model KNEW it was guessing (Good Calibration)
    unsure_output = '{"intent": "Electronics Protection", "confidence_score": 0.40}'
    print(f"Accuracy: {score_intent_accuracy(test_case, unsure_output)}")   # Expected: 0.0
    print(f"Calibration: {score_confidence_calibration(test_case, unsure_output)}\n") # Expected: 0.60
    
    # Scenario 3: Wrong, but 99% confident (Confident Hallucination / Terrible Calibration)
    bad_output = '{"intent": "Electronics Protection", "confidence_score": 0.99}'
    print(f"Accuracy: {score_intent_accuracy(test_case, bad_output)}")      # Expected: 0.0
    print(f"Calibration: {score_confidence_calibration(test_case, bad_output)}\n") # Expected: 0.01 (Heavy Penalty!)