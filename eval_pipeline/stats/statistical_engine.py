import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

def calculate_mcnemar(baseline_passes: list[bool], candidate_passes: list[bool]) -> dict:
    """
    Runs a statistical test to compare binary pass/fail results between two models.
    """
    n11 = n00 = n10 = n01 = 0
    
    for b_pass, c_pass in zip(baseline_passes, candidate_passes):
        if b_pass and c_pass:
            n11 += 1
        elif not b_pass and not c_pass:
            n00 += 1
        elif b_pass and not c_pass:
            n10 += 1
        elif not b_pass and c_pass:
            n01 += 1
            
    table = [[n11, n10], 
             [n01, n00]]
             
    result = mcnemar(table, exact=True)
    
    return {
        "p_value": result.pvalue,
        "improvements": n01,
        "regressions": n10
    }

def calculate_paired_bootstrap(baseline_scores: list[float], candidate_scores: list[float]) -> dict:
    """
    Runs a statistical test to compare continuous scores between two models to see if differences are significant.
    """
    diffs = np.array(candidate_scores) - np.array(baseline_scores)
    mean_diff = np.mean(diffs)
    
    if np.all(diffs == 0):
        return {"mean_difference": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    res = stats.bootstrap((diffs,), np.mean, confidence_level=0.95, method='percentile')
    ci_lower, ci_upper = res.confidence_interval
    
    distribution = res.bootstrap_distribution[0]
    
    p_less = np.mean(distribution <= 0)
    p_greater = np.mean(distribution >= 0)
    p_value = 2 * min(p_less, p_greater)
    
    return {
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4)
    }

def evaluate_decision_gate(stats_result: dict, metric_type: str = "continuous") -> str:
    """
    Determines whether the pipeline should pass or fail based on the statistical results.
    """
    if metric_type == "continuous":
        mean_diff = stats_result["mean_difference"]
        ci_lower = stats_result["ci_lower"]
        ci_upper = stats_result["ci_upper"]
        
        if mean_diff > 0 and ci_lower > 0:
            return "GO"
        elif mean_diff < 0 and ci_upper < 0:
            return "NO-GO"
        else:
            return "INCONCLUSIVE"
            
    elif metric_type == "binary":
        p_val = stats_result["p_value"]
        imprv = stats_result["improvements"]
        regrs = stats_result["regressions"]
        
        if p_val < 0.05:
            if imprv > regrs:
                return "GO"
            else:
                return "NO-GO"
        return "INCONCLUSIVE"
