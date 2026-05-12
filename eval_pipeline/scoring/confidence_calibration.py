from typing import List, Dict

def calculate_ece(predictions: List[Dict], num_bins: int = 10) -> float:
    """
    Calculates the Expected Calibration Error (ECE).
    Predictions must be a list of dicts: [{'is_correct': bool, 'confidence': float}]
    """
    if not predictions:
        return 0.0

    # 1. Initialize Bins (e.g., 10 bins: 0.0-0.1, 0.1-0.2... 0.9-1.0)
    bins = {i: {'correct_count': 0, 'total_count': 0, 'sum_confidence': 0.0} for i in range(num_bins)}
    
    # 2. Assign predictions to bins
    for pred in predictions:
        conf = pred['confidence']
        # Handle edge case where confidence is exactly 1.0 (put in the last bin)
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        
        bins[bin_idx]['total_count'] += 1
        bins[bin_idx]['sum_confidence'] += conf
        if pred['is_correct']:
            bins[bin_idx]['correct_count'] += 1

    # 3. Calculate ECE
    n_total = len(predictions)
    ece = 0.0
    
    for bin_data in bins.values():
        bin_total = bin_data['total_count']
        if bin_total == 0:
            continue # Skip empty bins
            
        # Calculate Accuracy and average Confidence for this bin
        bin_acc = bin_data['correct_count'] / bin_total
        bin_conf = bin_data['sum_confidence'] / bin_total
        
        # Add to ECE using the formula: |Bm|/N * |acc(Bm) - conf(Bm)|
        weight = bin_total / n_total
        ece += weight * abs(bin_acc - bin_conf)
        
    return ece

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Confidence Calibration (ECE)...\n")
    
    # Scenario 1: A perfectly calibrated model
    # It is 90% confident when it's right, and 50% confident when it's basically guessing.
    perfect_model = [
        {'is_correct': True, 'confidence': 0.95},
        {'is_correct': True, 'confidence': 0.90},
        {'is_correct': True, 'confidence': 0.85},
        {'is_correct': False, 'confidence': 0.40}, # Missed it, but knew it was unsure
        {'is_correct': False, 'confidence': 0.45},
    ]
    
    ece_perfect = calculate_ece(perfect_model)
    print(f"Perfectly Calibrated Model:")
    print(f" -> ECE: {round(ece_perfect, 4)} (Closer to 0 is better)\n")
    
    # Scenario 2: A dangerously overconfident model (Hallucinating)
    # It gets things wrong, but claims 99% confidence anyway.
    dangerous_model = [
        {'is_correct': True, 'confidence': 0.95},
        {'is_correct': True, 'confidence': 0.90},
        {'is_correct': True, 'confidence': 0.85},
        {'is_correct': False, 'confidence': 0.99}, # WRONG, but 99% confident!
        {'is_correct': False, 'confidence': 0.95}, # WRONG, but 95% confident!
    ]
    
    ece_dangerous = calculate_ece(dangerous_model)
    print(f"Dangerously Overconfident Model:")
    print(f" -> ECE: {round(ece_dangerous, 4)} (High penalty for overconfidence!)")