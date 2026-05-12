import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# ==========================================
# 1. McNEMAR'S TEST (For Binary Pass/Fail Metrics)
# ==========================================
def calculate_mcnemar(baseline_passes: list[bool], candidate_passes: list[bool]) -> dict:
    """
    Executes McNemar's test for paired binary data.
    """
    n11 = n00 = n10 = n01 = 0
    
    for b_pass, c_pass in zip(baseline_passes, candidate_passes):
        if b_pass and c_pass:
            n11 += 1 # Both passed
        elif not b_pass and not c_pass:
            n00 += 1 # Both failed
        elif b_pass and not c_pass:
            n10 += 1 # Regression (Baseline passed, Candidate failed)
        elif not b_pass and c_pass:
            n01 += 1 # Improvement (Baseline failed, Candidate passed)
            
    # Construct 2x2 Contingency Table
    table = [[n11, n10], 
             [n01, n00]]
             
    # Calculate exact p-value
    result = mcnemar(table, exact=True)
    
    return {
        "p_value": result.pvalue,
        "improvements": n01,
        "regressions": n10
    }

# ==========================================
# 2. PAIRED BOOTSTRAP (For Continuous Metrics 0.0 - 1.0)
# ==========================================
def calculate_paired_bootstrap(baseline_scores: list[float], candidate_scores: list[float]) -> dict:
    """
    Executes 10,000 paired bootstrap resamples to find the 95% Confidence Interval
    and calculates the empirical p-value.
    """
    # Calculate the raw differences between the paired scores
    diffs = np.array(candidate_scores) - np.array(baseline_scores)
    mean_diff = np.mean(diffs)
    
    # If all differences are exactly 0, scipy bootstrap will crash. Handle edge case:
    if np.all(diffs == 0):
        return {"mean_difference": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    # Execute Paired Bootstrap Resampling
    res = stats.bootstrap((diffs,), np.mean, confidence_level=0.95, method='percentile')
    ci_lower, ci_upper = res.confidence_interval
    
    # Extract the 10,000 resampled means
    distribution = res.bootstrap_distribution[0]
    
    # Calculate empirical 2-sided p-value
    # It counts what fraction of the 10,000 resamples crossed the 0 line
    p_less = np.mean(distribution <= 0)
    p_greater = np.mean(distribution >= 0)
    p_value = 2 * min(p_less, p_greater)
    
    return {
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4)  # <-- The new p-value!
    }

# ==========================================
# 3. THE GO / NO-GO DECISION MATRIX
# ==========================================
def evaluate_decision_gate(stats_result: dict, metric_type: str = "continuous") -> str:
    """
    Applies GrabOn's strict CI/CD blocking rules.
    """
    if metric_type == "continuous":
        mean_diff = stats_result["mean_difference"]
        ci_lower = stats_result["ci_lower"]
        ci_upper = stats_result["ci_upper"]
        
        if mean_diff > 0 and ci_lower > 0:
            return "GO"          # Statistically significant improvement
        elif mean_diff < 0 and ci_upper < 0:
            return "NO-GO"       # Statistically significant regression
        else:
            return "INCONCLUSIVE" # 95% CI crosses zero (just random noise)
            
    elif metric_type == "binary":
        p_val = stats_result["p_value"]
        imprv = stats_result["improvements"]
        regrs = stats_result["regressions"]
        
        if p_val < 0.05:
            if imprv > regrs:
                return "GO"      # Significant improvement
            else:
                return "NO-GO"   # Significant regression
        return "INCONCLUSIVE"    # Not enough evidence to prove it wasn't luck

# ==========================================
# TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Paired Bootstrap Engine...")
    base_scores = [0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7] * 3 # 30 samples
    cand_scores = [0.9, 0.8, 0.9, 0.9, 0.7, 0.8, 0.9, 0.9, 0.9, 0.8] * 3 # Generally better
    
    result = calculate_paired_bootstrap(base_scores, cand_scores)
    decision = evaluate_decision_gate(result, "continuous")
    
    print(f"Mean Difference: {result['mean_difference']}")
    print(f"95% CI: [{result['ci_lower']}, {result['ci_upper']}]")
    print(f"CI/CD Pipeline Action: {decision}")