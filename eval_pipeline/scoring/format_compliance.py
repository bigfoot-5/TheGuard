def score_format_compliance(generated_text: str, expected_max_length: int) -> float:
    """
    Evaluates if the generated deal copy adheres to the strict channel character limits.
    Returns a continuous score between 0.0 and 1.0 using a step-decay penalty.
    """
    # 1. Clean the text (remove accidental leading/trailing whitespace)
    cleaned_text = generated_text.strip()
    actual_length = len(cleaned_text)
    
    # 2. Perfect Compliance: Under or exactly at the limit
    if actual_length <= expected_max_length:
        return 1.0
        
    # 3. Calculate how far over the limit it is (as a percentage)
    percent_over = (actual_length - expected_max_length) / expected_max_length
    
    # 4. Step-Decay Logic
    if percent_over <= 0.10:
        # Up to 10% over the limit (e.g., 176 chars for WhatsApp's 160 limit)
        # Minor penalty, output might still be usable with a quick manual edit
        return 0.8
        
    elif percent_over <= 0.25:
        # Up to 25% over the limit (e.g., 200 chars for WhatsApp)
        # Heavy penalty, requires significant rewriting
        return 0.5
        
    else:
        # More than 25% over the limit
        # Complete failure, useless for the target channel
        return 0.0

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Format Compliance Scoring...")
    
    # WhatsApp Limit: 160 characters
    wa_limit = 160
    
    # Test 1: Perfect length (130 chars)
    perfect_copy = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight!"
    print(f"Perfect (<160): {score_format_compliance(perfect_copy, wa_limit)}")  # Expected: 1.0
    
    # Test 2: Slightly over (174 chars -> ~8% over)
    slight_over = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight! Don't miss out on this deal."
    print(f"Slightly Over (174): {score_format_compliance(slight_over, wa_limit)}")  # Expected: 0.8
    
    # Test 3: Moderately over (198 chars -> ~23% over)
    moderate_over = "Flash Sale! Get 50% off all electronics at GrabOn this weekend only. Use code FLASH50 at checkout. Hurry, offer ends Sunday at midnight! We have smartphones, laptops, and headphones on clearance."
    print(f"Moderately Over (198): {score_format_compliance(moderate_over, wa_limit)}")  # Expected: 0.5
    
    # Test 4: Complete failure (250+ chars)
    fail_copy = perfect_copy + " " + perfect_copy
    print(f"Total Failure (>200): {score_format_compliance(fail_copy, wa_limit)}")  # Expected: 0.0