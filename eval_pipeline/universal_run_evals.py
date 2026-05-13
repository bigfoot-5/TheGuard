import sys
import os
import json
import yaml
import subprocess
from datetime import datetime

# Import LLM Runner, Stats Engine, and our new Registry
from llm_runner import generate_response
from stats.statistical_engine import calculate_paired_bootstrap, evaluate_decision_gate, calculate_mcnemar
from metrics_registry import REGISTRY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- TELEMETRY & BASELINE STORAGE ---
def get_current_commit():
    try:
        if "PR_COMMIT_SHA" in os.environ: return os.environ["PR_COMMIT_SHA"][:7]
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown_commit"

def save_eval_results(averages_dict: dict, raw_arrays_dict: dict, provider: str, status: str):
    history_path = os.path.join(BASE_DIR, "data/history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    commit_hash = get_current_commit()
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f: history = json.load(f)
        except: pass

    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": "Full Config Suite", 
        "provider": provider,
        "commit_hash": commit_hash,
        "status": status, # <--- NEW: Tracks if this was a GO or NO-GO
        "averages": averages_dict,
        "raw_arrays": raw_arrays_dict
    }
    history.append(new_entry)
    with open(history_path, "w") as f: json.dump(history, f, indent=4)

def load_latest_baseline(current_raw_arrays: dict):
    """Loads the most recent SUCCESSFUL baseline that matches the exact sample size."""
    history_path = os.path.join(BASE_DIR, "data/history.json")
    if not os.path.exists(history_path): return None
    
    # Figure out how many samples we ran this time
    if not current_raw_arrays: return None
    expected_length = len(list(current_raw_arrays.values())[0])
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            if not data: return None
            
            # Look backwards to find the last SUCCESSFUL run...
            for entry in reversed(data):
                if entry.get("status", "") in ["GO (STABLE)", "GO (IMPROVED)", "GO (BASELINE)"]:
                    base_arrays = entry.get("raw_arrays", {})
                    if base_arrays:
                        # ...that also has the EXACT SAME sample size!
                        base_length = len(list(base_arrays.values())[0])
                        if base_length == expected_length:
                            return base_arrays
                            
            # If we looked through history and found no GO runs with this length,
            # we must trigger a Cold Start for this specific sample size.
            return None
    except:
        return None

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting Config-Driven Evaluation Pipeline...")
    
    # 1. LOAD YAML CONFIG
    config_path = os.path.join(PROJECT_ROOT, "eval_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_model = config["global"]["default_model"]
    
    current_raw_arrays = {}
    current_averages = {}
    metric_types = {} # Tracks if a metric is continuous or binary for the stats engine

    # 2. DYNAMIC EVALUATION LOOP
    for task_name, task_config in config["tasks"].items():
        print(f"\n⏳ Evaluating Task: {task_name}...")
        
        # Load dataset & prompt
        with open(os.path.join(PROJECT_ROOT, task_config["dataset"]), "r") as f: 
            cases = json.load(f)[:5] # Remove [:5] to run the full 30 cases
            
        with open(os.path.join(PROJECT_ROOT, task_config["prompt_file"]), "r") as f: 
            system_prompt = f.read()

        model = task_config.get("model", global_model)
        temp = task_config.get("temperature", config["global"]["default_temperature"])
        
        # Initialize arrays for the metrics tracked in this task
        for metric in task_config["metrics"]:
            reg_key = metric["registry_key"]
            current_raw_arrays[reg_key] = []
            metric_types[reg_key] = {"name": metric["name"], "type": metric["type"]}

        # Run Inference and Score
        for case in cases:
            # Dynamically build the prompt from the JSON keys
            user_input = task_config["input_template"].format(**case)
            output = generate_response(system_prompt, user_input, model=model, temperature=temp)
            
            # Dynamically run every metric listed in the YAML
            for metric in task_config["metrics"]:
                reg_key = metric["registry_key"]
                scoring_function = REGISTRY[reg_key]
                
                score = scoring_function(case, output)
                current_raw_arrays[reg_key].append(score)

    # Calculate averages for telemetry
    for key, array in current_raw_arrays.items():
        current_averages[key] = sum(array) / len(array) if array else 0.0

    # 3. STATISTICAL ENGINE & CI/CD GATE
    baseline_raw_arrays = load_latest_baseline(current_raw_arrays)
    
    if not baseline_raw_arrays:
        print("\n⚠️ COLD START: No previous baseline history found.")
        # Added the required status flag!
        save_eval_results(current_averages, current_raw_arrays, provider=global_model, status="GO (BASELINE)")
        print("✅ GO: Pipeline initialized.")
        sys.exit(0)
        
    print("\n📊 Executing Phase 5: Statistical Comparison Engine...")
    final_decision = "GO"
    failure_reasons = []

    for reg_key, data_type_info in metric_types.items():
        if reg_key not in baseline_raw_arrays: continue
            
        print(f"\nEvaluating: {data_type_info['name']}...")
        base_array = baseline_raw_arrays[reg_key]
        cand_array = current_raw_arrays[reg_key]
        
        if data_type_info["type"] == "continuous":
            stats_result = calculate_paired_bootstrap(base_array, cand_array)
            print(f"  -> 95% CI: [{stats_result['ci_lower']:+.4f}, {stats_result['ci_upper']:+.4f}]")
        else:
            stats_result = calculate_mcnemar(base_array, cand_array)
            print(f"  -> McNemar P-Value: {stats_result['p_value']:.4f}")
            
        metric_decision = evaluate_decision_gate(stats_result, metric_type=data_type_info["type"])
        
        if metric_decision == "NO-GO":
            final_decision = "NO-GO"
            
            # --- NEW: Identify exactly which test cases dropped in quality ---
            regressions = []
            for idx, (b_score, c_score) in enumerate(zip(base_array, cand_array)):
                if c_score < b_score: 
                    regressions.append(f"#{idx + 1}")
            
            print(f"  -> 🚨 Regression detected on Test Cases: {', '.join(regressions)}")
            failure_reasons.append(f"{data_type_info['name']} (Failed cases: {', '.join(regressions)})")
            
        elif metric_decision == "INCONCLUSIVE" and final_decision != "NO-GO":
            final_decision = "INCONCLUSIVE"

    # 5. THE MASTER GO / NO-GO GATE
    print("\n=========================================")
    if final_decision == "NO-GO":
        print(f"❌ CRITICAL NO-GO: PR Blocked. Reasons: {' | '.join(failure_reasons)}")
        save_eval_results(current_averages, current_raw_arrays, provider=global_model, status="NO-GO")
        sys.exit(1)
        
    elif final_decision == "INCONCLUSIVE":
        print(f"✅ GO (STABLE): No statistically significant difference detected. Safe to merge.")
        save_eval_results(current_averages, current_raw_arrays, provider=global_model, status="GO (STABLE)")
        sys.exit(0)
        
    else:
        print(f"✅ GO (IMPROVED): Pipeline detected statistically significant improvements!")
        save_eval_results(current_averages, current_raw_arrays, provider=global_model, status="GO (IMPROVED)")
        sys.exit(0)

if __name__ == "__main__":
    main()