import json

def _parse_llm_json(raw_text: str) -> dict:
    """
    Extracts JSON from the model output.
    """
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = raw_text[start_idx:end_idx+1]
            return json.loads(json_str)
            
        return {}
    except Exception:
        return {}

def score_intent_accuracy(case: dict, generated_text: str) -> float:
    """
    Checks if the model's predicted intent matches the correct intent.
    Returns 1.0 for a match, 0.0 otherwise.
    """
    expected_intent = case.get("expected_intent", "").strip().lower()
    
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    
    if predicted_intent == expected_intent and expected_intent != "":
        return 1.0
    return 0.0

def score_confidence_calibration(case: dict, generated_text: str) -> float:
    """
    Evaluates how well the model's confidence matches its actual accuracy.
    A score of 1.0 means perfect self-awareness; 0.0 means it confidently hallucinated.
    """
    expected_intent = case.get("expected_intent", "").strip().lower()
    
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    
    confidence = float(output_data.get("confidence_score", 0.0))
    is_correct = (predicted_intent == expected_intent)
    ideal_confidence = 1.0 if is_correct else 0.0
    error = abs(ideal_confidence - confidence)
    
    return 1.0 - error

def score_edge_case_handling(case: dict, generated_text: str) -> float:
    """
    Checks if the model safely handles unclear situations by either outputting 'No Insurance Applicable' or expressing low confidence.
    """
    is_edge_case = case.get("is_ambiguous_edge_case", False)
    
    if not is_edge_case:
        return 1.0 
        
    output_data = _parse_llm_json(generated_text)
    predicted_intent = output_data.get("intent", "").strip().lower()
    confidence = float(output_data.get("confidence_score", 1.0))
    
    if predicted_intent == "no insurance applicable":
        return 1.0
    elif confidence <= 0.60:
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    test_case = {"expected_intent": "Travel Cancellation"}
    
    good_output = '{"intent": "Travel Cancellation", "confidence_score": 0.95}'
    print(f"Accuracy: {score_intent_accuracy(test_case, good_output)}")
    print(f"Calibration: {score_confidence_calibration(test_case, good_output)}\n")
    
    unsure_output = '{"intent": "Electronics Protection", "confidence_score": 0.40}'
    print(f"Accuracy: {score_intent_accuracy(test_case, unsure_output)}")
    print(f"Calibration: {score_confidence_calibration(test_case, unsure_output)}\n")
    
    bad_output = '{"intent": "Electronics Protection", "confidence_score": 0.99}'
    print(f"Accuracy: {score_intent_accuracy(test_case, bad_output)}")
    print(f"Calibration: {score_confidence_calibration(test_case, bad_output)}\n")