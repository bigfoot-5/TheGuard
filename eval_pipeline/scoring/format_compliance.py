def score_format_compliance(case: dict, generated_text: str) -> float:
    """
    Checks if the generated text is under the character limit.
    Returns a score from 1.0 (perfect) down to 0.0 depending on how much it goes over the limit.
    """
    expected_max_length = case.get("expected_max_length", 160)
    
    cleaned_text = generated_text.strip()
    actual_length = len(cleaned_text)
    
    if actual_length <= expected_max_length:
        return 1.0
        
    percent_over = (actual_length - expected_max_length) / expected_max_length
    if percent_over <= 0.10:
        return 0.8
        
    elif percent_over <= 0.25:
        return 0.5
        
    else:
        return 0.0

if __name__ == "__main__":
    mock_case = {"expected_max_length": 160}
    
    perfect_copy = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight!"
    print(f"Perfect (<160): {score_format_compliance(mock_case, perfect_copy)}")
    
    slight_over = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight! Don't miss out on this deal."
    print(f"Slightly Over (174): {score_format_compliance(mock_case, slight_over)}")
    
    moderate_over = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight! We have smartphones, laptops, and headphones on clearance."
    print(f"Moderately Over (198): {score_format_compliance(mock_case, moderate_over)}")
    
    fail_copy = perfect_copy + " " + perfect_copy
    print(f"Total Failure (>200): {score_format_compliance(mock_case, fail_copy)}")
