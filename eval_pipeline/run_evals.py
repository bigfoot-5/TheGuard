import sys
import os
import json
from datetime import datetime

# Import LLM Runner
from llm_runner import generate_response

# Import all 5 Scoring Metrics
from scoring.format_compliance import score_format_compliance
from scoring.semantic_similarity import score_semantic_similarity
from scoring.llm_judge import score_persuasiveness, score_factual_grounding
from scoring.confidence_calibration import calculate_ece, score_insurance_intent

# --- TELEMETRY STORAGE ---
def save_eval_results(task_name: str, scores_dict: dict, provider: str = "gpt-4o-mini", version: str = "latest"):
    history_path = "data/history.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try: history = json.load(f)
            except: history = []
    else: history = []

    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task_name, "version": version, "provider": provider,
        **scores_dict
    }
    history.append(new_entry)
    with open(history_path, "w") as f: json.dump(history, f, indent=4)

# --- EVALUATOR 1: DEAL COPY ---
def evaluate_deal_copy(model: str):
    print(f"\n[1/3] ⏳ Evaluating Deal Copy against {model}...")
    with open("data/deal_copy_cases.json", "r") as f: cases = json.load(f)[:5] # Running 5 for demo speed
    
    scores = {"compliance": [], "similarity": [], "persuasiveness": []}
    prompt = "You are a marketing assistant. Write short, punchy deal copy."
    
    for case in cases:
        user_input = f"Write deal copy for: {case['raw_deal_extraction']} at {case['merchant_name']}. Target Channel: {case['target_channel']}."
        output = generate_response(prompt, user_input, model=model)
        
        scores["compliance"].append(score_format_compliance(output, case['expected_max_length']))
        scores["similarity"].append(score_semantic_similarity(case['raw_deal_extraction'], output))
        scores["persuasiveness"].append(score_persuasiveness(output))
        
    return {k: sum(v)/len(v) for k, v in scores.items()}

# --- EVALUATOR 2: CREDIT NARRATIVE ---
def evaluate_credit_narrative(model: str):
    print(f"\n[2/3] ⏳ Evaluating Credit Faithfulness against {model}...")
    with open("data/credit_narrative_cases.json", "r") as f: cases = json.load(f)[:5]
    
    grounding_scores = []
    prompt = "You are an underwriter. Summarize the merchant data factually. Do not invent numbers."
    
    for case in cases:
        user_input = f"Write a risk summary for merchant: {json.dumps(case['merchant_data'])}"
        output = generate_response(prompt, user_input, model=model)
        
        # Metric: Factual Grounding (NLI)
        grounding = score_factual_grounding(case['merchant_data'], output)
        grounding_scores.append(grounding)
        
    return {"grounding": sum(grounding_scores) / len(grounding_scores)}

# --- EVALUATOR 3: INSURANCE INTENT ---
def evaluate_insurance_intent(model: str):
    print(f"\n[3/3] ⏳ Evaluating Insurance ECE against {model}...")
    with open("data/insurance_intent_cases.json", "r") as f: cases = json.load(f)[:5]
    
    predictions = []
    prompt = "Classify this cart into one insurance category. Respond ONLY with JSON: {'intent': 'Category Name', 'confidence': 0.95}"
    
    for case in cases:
        user_input = f"Cart contents: {case['cart_contents']}. Categories: {case['available_categories']}"
        output_str = generate_response(prompt, user_input, model=model, temperature=0.1)
        
        try:
            output_json = json.loads(output_str.strip('` \n').replace('json\n', ''))
            is_correct = (output_json.get("intent") == case['expected_intent'])
            confidence = float(output_json.get("confidence", 0.5))
        except:
            is_correct, confidence = False, 1.0 
            
        predictions.append({'is_correct': is_correct, 'confidence': confidence})
        
    # FIX: Actually use the scoring function we built!
    blended_intent_score = score_insurance_intent(predictions)
    
    # We still calculate raw ECE just to pass it to the Streamlit Dashboard
    raw_ece = calculate_ece(predictions) 
    
    return {"intent_score": blended_intent_score, "ece": raw_ece}

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting GrabOn CI/CD Evaluation Pipeline...")
    test_model = "gpt-4o-mini"
    
    # Run all 3 Datasets
    deal_metrics = evaluate_deal_copy(test_model)
    credit_metrics = evaluate_credit_narrative(test_model)
    insurance_metrics = evaluate_insurance_intent(test_model)
    
    # Combine results
    final_metrics = {**deal_metrics, **credit_metrics, **insurance_metrics}
    
    print("\n📊 Final Aggregated Results:")
    print(json.dumps(final_metrics, indent=2))
    
    # Save to history for Streamlit App
    save_eval_results("Full Suite", final_metrics, provider=test_model)
    
    # Decision Gate
    # Require 90% format compliance AND 90% factual grounding
    decision = "GO" if final_metrics.get("compliance", 0) >= 0.90 and final_metrics.get("grounding", 0) >= 0.90 else "NO-GO"
    
    if decision == "NO-GO":
        print("❌ CRITICAL: Pipeline detected a regression. Blocking PR.")
        sys.exit(1)
    else:
        print("✅ Pipeline passed all checks. GO for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()