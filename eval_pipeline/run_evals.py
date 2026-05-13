from datetime import time
import sys
import os
import json
from datetime import datetime
import time as timer
import subprocess

# Import LLM Runner
from llm_runner import generate_response

# Import all 5 Scoring Metrics
from scoring.format_compliance import score_format_compliance
from scoring.semantic_similarity import score_semantic_similarity
from scoring.llm_judge import score_persuasiveness, score_factual_grounding
from scoring.confidence_calibration import calculate_ece
from stats.statistical_engine import calculate_paired_bootstrap, evaluate_decision_gate, calculate_mcnemar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- TELEMETRY & BASELINE STORAGE ---
def get_current_commit():
    try:
        # Grabs the short 7-character Git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
        return commit_hash
    except Exception:
        return "unknown_commit"

def save_eval_results(task_name: str, averages_dict: dict, raw_arrays_dict: dict, provider: str = "gpt-4o-mini"):
    history_path = os.path.join(BASE_DIR, "data/history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    commit_hash = get_current_commit()
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            try: history = json.load(f)
            except: history = []
    else: history = []

    # We save BOTH averages (for the dashboard) and raw arrays (for the stats engine)
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task_name, 
        "provider": provider,
        "commit_hash": commit_hash,
        "averages": averages_dict,
        "raw_arrays": raw_arrays_dict
    }
    history.append(new_entry)
    with open(history_path, "w") as f: json.dump(history, f, indent=4)

def load_latest_baseline():
    """Loads the most recent raw arrays from history.json to use as the baseline."""
    history_path = os.path.join(BASE_DIR, "data/history.json")
    
    if not os.path.exists(history_path): 
        return None
        
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            
            # If the file is valid JSON but empty, treat it as no baseline
            if not data or len(data) == 0:
                return None
                
            # Return ONLY the raw arrays from the very last successful run!
            return data[-1].get("raw_arrays")
            
    except json.JSONDecodeError:
        # If the file is completely blank (0 bytes) or corrupted, catch the crash
        print("⚠️ WARNING: history.json is corrupted or completely blank.")
        return None

# --- EVALUATORS ---
def evaluate_deal_copy(model: str):
    print(f"\n[1/3] ⏳ Evaluating Deal Copy against {model}...")
    with open(os.path.join(BASE_DIR, "data/deal_copy_cases.json"), "r") as f: cases = json.load(f)[:5]
    try:
        with open(os.path.join(BASE_DIR, "prompts/system_assistant_389c7a3e.txt"), "r") as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = "You are a marketing assistant. Write short, punchy deal copy."

    scores = {"compliance": [], "similarity": [], "persuasiveness": []}
    for case in cases:
        user_input = f"Write deal copy for: {case['raw_deal_extraction']} at {case['merchant_name']}. Target Channel: {case['target_channel']}."
        output = generate_response(prompt, user_input, model=model)
        scores["compliance"].append(score_format_compliance(output, case['expected_max_length']))
        scores["similarity"].append(score_semantic_similarity(case['raw_deal_extraction'], output))
        # scores["persuasiveness"].append(score_persuasiveness(output))
    timer.sleep(65)
    return scores

def evaluate_credit_narrative(model: str):
    print(f"\n[2/3] ⏳ Evaluating Credit Faithfulness against {model}...")
    with open(os.path.join(BASE_DIR, "data/credit_narrative_cases.json"), "r") as f: cases = json.load(f)[:5]
    grounding_scores = []
    try:
        with open(os.path.join(BASE_DIR, "prompts/system_underwriter_a93040d0.txt"), "r") as f: prompt = f.read()
    except FileNotFoundError:
        prompt = "You are an underwriter. Summarize the merchant data factually. Do not invent numbers."
    for case in cases:
        # Pass the whole 'case' dictionary
        user_input = f"Write a risk summary for merchant: {json.dumps(case)}"
        output = generate_response(prompt, user_input, model=model)
        grounding_scores.append(score_factual_grounding(case, output))
    return {"grounding": grounding_scores}

def evaluate_insurance_intent(model: str):
    print(f"\n[3/3] ⏳ Evaluating Insurance Intent against {model}...")
    with open(os.path.join(BASE_DIR, "data/insurance_intent_cases.json"), "r") as f: cases = json.load(f)[:5]
    predictions = []
    try:
        with open(os.path.join(BASE_DIR, "prompts/system_insurance_classifier_b5a8b780.txt"), "r") as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = "Classify this cart into one insurance category. Respond ONLY with JSON: {'intent': 'Category Name', 'confidence': 0.95}"
        
    allowed_categories = [
        "Electronics Damage Protection", 
        "Travel Cancellation", 
        "Fraud/Cyber Protection", 
        "Health/Accident", 
        "No Insurance Applicable"
    ]
    
    for case in cases:
        user_input = f"Cart contents: {case['cart_contents']}. Device: {case['user_device_type']}. History: {case['historical_purchase_flags']}. Allowed Categories: {allowed_categories}"
        
        output_str = generate_response(prompt, user_input, model=model, temperature=0.1)
        
        try:
            output_json = json.loads(output_str.strip('` \n').replace('json\n', ''))
            is_correct = (output_json.get("intent") == case['expected_intent'])
            confidence = float(output_json.get("confidence", 0.5))
        except:
            is_correct, confidence = False, 1.0 
            
        predictions.append({'is_correct': is_correct, 'confidence': confidence})
        
    intent_scores = [1.0 if p['is_correct'] else 0.0 for p in predictions]
    raw_ece = calculate_ece(predictions) 
    
    return {"intent_score": intent_scores, "ece": raw_ece}

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting GrabOn CI/CD Evaluation Pipeline...")
    test_model = "gpt-4o-mini"
    
    # 1. RUN THE CANDIDATE MODEL
    candidate_deal = evaluate_deal_copy(test_model)
    # candidate_credit = evaluate_credit_narrative(test_model)
    candidate_insurance = evaluate_insurance_intent(test_model)
    
    # 2. COMPILE CURRENT RUN DATA
    current_raw_arrays = {
        "similarity": candidate_deal["similarity"],
        # "persuasiveness": candidate_deal["persuasiveness"],
        # "grounding": candidate_credit["grounding"],
        "compliance": candidate_deal["compliance"],
        "intent_score": candidate_insurance["intent_score"]  
    }
    
    current_averages = {
        "compliance": sum(candidate_deal["compliance"]) / len(candidate_deal["compliance"]),
        "similarity": sum(candidate_deal["similarity"]) / len(candidate_deal["similarity"]),
        # "persuasiveness": sum(candidate_deal["persuasiveness"]) / len(candidate_deal["persuasiveness"]),
        # "grounding": sum(candidate_credit["grounding"]) / len(candidate_credit["grounding"]),
        "intent_accuracy": sum(candidate_insurance["intent_score"]) / len(candidate_insurance["intent_score"]),
        "ece": candidate_insurance["ece"]
    }

    # 3. LOAD BASELINE (The "Cold Start" Check)
    baseline_raw_arrays = load_latest_baseline()
    
    if not baseline_raw_arrays:
        print("\n⚠️ COLD START: No previous baseline history found.")
        print("💾 Saving current run as the initial production baseline...")
        save_eval_results("Full Suite", current_averages, current_raw_arrays, provider=test_model)
        print("✅ GO: Pipeline initialized. Safe for deployment.")
        sys.exit(0)
        
    # 4. STATISTICAL COMPARISON ENGINE 
    print("\n📊 Executing Phase 5: Statistical Comparison Engine...")
    
    # We map every metric, its data arrays, AND its statistical type
    metrics_to_test = {
        "Semantic Similarity": {"baseline": baseline_raw_arrays["similarity"], "candidate": current_raw_arrays["similarity"], "type": "continuous"},
        # "Persuasiveness": {"baseline": baseline_raw_arrays["persuasiveness"], "candidate": current_raw_arrays["persuasiveness"], "type": "continuous"},
        # "Factual Grounding": {"baseline": baseline_raw_arrays["grounding"], "candidate": current_raw_arrays["grounding"], "type": "continuous"},
        "Format Compliance": {"baseline": baseline_raw_arrays["compliance"], "candidate": current_raw_arrays["compliance"], "type": "binary"},
        "Intent Accuracy": {"baseline": baseline_raw_arrays["intent_score"], "candidate": current_raw_arrays["intent_score"], "type": "binary"}
    }

    final_decision = "GO"
    failure_reasons = []

    for metric_name, data in metrics_to_test.items():
        print(f"\nEvaluating: {metric_name} ({data['type']} metric)...")
        
        # Route to the correct statistical test
        if data["type"] == "continuous":
            stats_result = calculate_paired_bootstrap(data["baseline"], data["candidate"])
            print(f"  -> 95% CI: [{stats_result['ci_lower']:+.4f}, {stats_result['ci_upper']:+.4f}]")
            print(f"  -> P-Value: {stats_result.get('p_value', 'N/A')}")
        else:
            stats_result = calculate_mcnemar(data["baseline"], data["candidate"])
            print(f"  -> McNemar P-Value: {stats_result['p_value']:.4f}")
            print(f"  -> Improvements: {stats_result['improvements']} | Regressions: {stats_result['regressions']}")
        
        # Pass the type into the decision gate so it knows how to read the result
        metric_decision = evaluate_decision_gate(stats_result, metric_type=data["type"])
        
        if metric_decision == "NO-GO":
            final_decision = "NO-GO"
            failure_reasons.append(f"{metric_name} regressed")
        elif metric_decision == "INCONCLUSIVE" and final_decision != "NO-GO":
            final_decision = "INCONCLUSIVE"

    # 5. THE MASTER GO / NO-GO GATE
    print("\n=========================================")
    if final_decision == "NO-GO":
        print(f"❌ CRITICAL NO-GO: PR Blocked.")
        print(f"Reasons: {', '.join(failure_reasons)}")
        sys.exit(1)
    elif final_decision == "INCONCLUSIVE":
        print(f"⚠️ INCONCLUSIVE: Changes are statistical noise. Blocking PR to prevent churn.")
        sys.exit(1)
    else:
        print(f"✅ GO: All metrics show statistically significant improvement or stable performance.")
        print("💾 Saving new run as the updated production baseline...")
        # Only save to history if it actually passed! We don't want a regression becoming the new baseline.
        save_eval_results("Full Suite", current_averages, current_raw_arrays, provider=test_model)
        print("🚀 Safe for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    main()