import sys
import os
import json
import yaml
import subprocess
from datetime import datetime
import time
import csv

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

def save_eval_results(averages_dict: dict, raw_arrays_dict: dict, provider: str, status: str, failed_cases: dict = None, cost_data: dict = None, latency: float = 0.0, task_providers: dict = None):
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
        "status": status,
        "latency": latency,
        "cost_data": cost_data or {},
        "failed_cases": failed_cases or {},
        "averages": averages_dict,
        "raw_arrays": raw_arrays_dict,
        "task_providers": task_providers or {}
    }
    history.append(new_entry)
    with open(history_path, "w") as f: json.dump(history, f, indent=4)

def load_latest_baseline(current_raw_arrays: dict):
    """Loads the most recent SUCCESSFUL baseline that matches the exact sample size."""
    history_path = os.path.join(BASE_DIR, "data/history.json")
    if not os.path.exists(history_path): return None
    
    if not current_raw_arrays: return None
    expected_length = len(list(current_raw_arrays.values())[0])
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            if not data: return None
            
            for entry in reversed(data):
                if entry.get("status", "") in ["GO (STABLE)", "GO (IMPROVED)", "GO (BASELINE)"]:
                    base_arrays = entry.get("raw_arrays", {})
                    if base_arrays:
                        base_length = len(list(base_arrays.values())[0])
                        if base_length == expected_length:
                            return base_arrays
            return None
    except:
        return None

# --- MAIN ORCHESTRATOR ---
def main():
    print("🚀 Starting Config-Driven Evaluation Pipeline...")
    pipeline_start_time = time.time()
    
    # 1. LOAD YAML CONFIG
    config_path = os.path.join(PROJECT_ROOT, "eval_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_model = config["global"]["default_model"]
    
    current_raw_arrays = {}
    current_averages = {}
    metric_types = {} 
    metric_case_ids = {}

    # 2. DYNAMIC EVALUATION LOOP
    total_pipeline_cost = 0.0
    task_costs_dict = {}
    raw_csv_data = []
    task_providers_dict = {}
    
    for task_name, task_config in config["tasks"].items():
        print(f"\n⏳ Evaluating Task: {task_name}...")
        task_cost = 0.0
        
        # Load dataset & prompt (Ensure NO [:5] here!)
        with open(os.path.join(PROJECT_ROOT, task_config["dataset"]), "r") as f: 
            cases = json.load(f)
            
        task_ids = [
            c.get("deal_id") or c.get("cart_id") or c.get("merchant_id") or f"#{i+1}" 
            for i, c in enumerate(cases)
        ]

        with open(os.path.join(PROJECT_ROOT, task_config["prompt_file"]), "r") as f: 
            system_prompt = f.read()

        model = task_config.get("model", global_model)
        temp = task_config.get("temperature", config["global"]["default_temperature"])
        task_providers_dict[task_name] = model

        # Initialize arrays for the metrics tracked in this task
        for metric in task_config["metrics"]:
            reg_key = metric["registry_key"]
            current_raw_arrays[reg_key] = []
            metric_types[reg_key] = {"name": metric["name"], "type": metric["type"]}
            metric_case_ids[reg_key] = task_ids

        # Run Inference and Score
        for case in cases:
            case_start_time = time.time() 
            
            user_input = task_config["input_template"].format(**case)
            case_id = case.get("deal_id") or case.get("cart_id") or case.get("merchant_id") or "Unknown"
            
            csv_row = {
                "Task": task_name,
                "Case_ID": case_id,
                "Model_Used": model
            }
            
            # 1. Catch the generation
            output, gen_cost = generate_response(system_prompt, user_input, model=model, temperature=temp)
            case_cost = gen_cost
            
            for metric in task_config["metrics"]:
                reg_key = metric["registry_key"]
                scoring_function = REGISTRY[reg_key]
                
                # 2. Execute the scoring function
                result = scoring_function(case, output, **metric)
                
                # 3. Handle Smart Tuples
                if isinstance(result, tuple):
                    score = result[0]
                    eval_cost = result[1]
                else:
                    score = result
                    eval_cost = 0.0
                    
                csv_row[metric["name"]] = score
                case_cost += eval_cost
                current_raw_arrays[reg_key].append(score)
            
            csv_row["Total_Cost_USD"] = round(case_cost, 5)
            csv_row["Latency_Seconds"] = round(time.time() - case_start_time, 2)
            raw_csv_data.append(csv_row)
            
            task_cost += case_cost
            if "gemini" in model.lower() or "llama" in model.lower():
                time.sleep(2) # Rate limit protection

        task_costs_dict[task_name] = task_cost
        total_pipeline_cost += task_cost
        print(f"💰 Task '{task_name}' Cost: ${task_cost:.5f}")

    task_costs_dict["Total"] = total_pipeline_cost
    print(f"\n💸 TOTAL PIPELINE GENERATION COST: ${total_pipeline_cost:.5f}")
    
    # Calculate averages for telemetry
    for key, array in current_raw_arrays.items():
        current_averages[key] = sum(array) / len(array) if array else 0.0
        
    used_models = set([task.get("model", global_model) for task in config["tasks"].values()])
    actual_providers = " & ".join(used_models)

    # 3. STATISTICAL ENGINE & CI/CD GATE
    baseline_raw_arrays = load_latest_baseline(current_raw_arrays)
    total_execution_time = time.time() - pipeline_start_time
    
    if not baseline_raw_arrays:
        print("\n⚠️ COLD START: No previous baseline history found.")
        save_eval_results(current_averages, current_raw_arrays, provider=actual_providers, status="GO (BASELINE)", cost_data=task_costs_dict, latency=total_execution_time, task_providers=task_providers_dict)
        print("✅ GO: Pipeline initialized.")
        sys.exit(0)
        
    print("\n📊 Executing Phase 5: Statistical Comparison Engine...")
    final_decision = "GO"
    failure_reasons = []
    failed_cases_dict = {}

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
            ids = metric_case_ids[reg_key]
            regressions = [ids[idx] for idx, (b, c) in enumerate(zip(base_array, cand_array)) if c < b]
            failed_cases_dict[data_type_info['name']] = regressions 
            
            print(f"  -> 🚨 Regression detected on Test Cases: {', '.join(regressions)}")
            failure_reasons.append(f"{data_type_info['name']} (Failed cases: {', '.join(regressions)})")
            
        elif metric_decision == "INCONCLUSIVE" and final_decision != "NO-GO":
            final_decision = "INCONCLUSIVE"

    # ==========================================
    # EXPORT RAW CSV REPORT (ENHANCED WITH REGRESSIONS)
    # ==========================================
    for row in raw_csv_data:
        row["Regression_Detected"] = "False"
        for metric_name in row.keys():
            if metric_name in failed_cases_dict:
                if row["Case_ID"] in failed_cases_dict[metric_name]:
                    row["Regression_Detected"] = "True"
    
    csv_path = os.path.join(PROJECT_ROOT, "eval_report_raw.csv")
    fieldnames = ["Task", "Case_ID", "Model_Used", "Regression_Detected", "Total_Cost_USD", "Latency_Seconds"]
    for row in raw_csv_data:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
                
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_csv_data)
        
    print(f"📄 Generated final raw eval report: {csv_path}")

    # 5. THE MASTER GO / NO-GO GATE
    print("\n=========================================")
    print(f"⏱️ Total Pipeline Execution Time: {total_execution_time:.1f}s")
    
    if final_decision == "NO-GO":
        print(f"❌ CRITICAL NO-GO: PR Blocked. Reasons: {' | '.join(failure_reasons)}")
        save_eval_results(current_averages, current_raw_arrays, provider=actual_providers, status="NO-GO", failed_cases=failed_cases_dict, cost_data=task_costs_dict, latency=total_execution_time, task_providers=task_providers_dict)
        sys.exit(1)
        
    elif final_decision == "INCONCLUSIVE":
        print(f"✅ GO (STABLE): No statistically significant difference detected. Safe to merge.")
        save_eval_results(current_averages, current_raw_arrays, provider=actual_providers, status="GO (STABLE)", cost_data=task_costs_dict, latency=total_execution_time, task_providers=task_providers_dict)
        sys.exit(0)
        
    else:
        print(f"✅ GO (IMPROVED): Pipeline detected statistically significant improvements!")
        save_eval_results(current_averages, current_raw_arrays, provider=actual_providers, status="GO (IMPROVED)", cost_data=task_costs_dict, latency=total_execution_time, task_providers=task_providers_dict)
        sys.exit(0)

if __name__ == "__main__":
    main()