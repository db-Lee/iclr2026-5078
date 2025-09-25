import os
import json
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, average_precision_score

def compute_verification_metrics(predicted_labels, true_labels, scores):
    """Compute F1 and AUPRC for verification task"""
    if len(predicted_labels) == 0 or len(true_labels) == 0:
        return 0.0, 0.0
    
    f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
    
    # Calculate AUPRC
    if len(set(true_labels)) <= 1:
        auprc = 0.0
    else:
        auprc = average_precision_score(true_labels, scores)
    
    return f1, auprc

def get_reward_value(rewards, strategy):
    """Get reward value using specified strategy"""
    rewards = np.array(rewards)
    rewards[np.isnan(rewards)] = 0.5
    
    if strategy == "min":
        return np.min(rewards).item()
    elif strategy == "max":
        return np.max(rewards).item()
    elif strategy == "mean":
        return np.mean(rewards).item()
    else:  # strategy == "last"
        return rewards[-1]

def get_verification_prediction(rewards, strategy, threshold=0.5):
    """Get verification prediction from rewards using specified strategy"""
    score = get_reward_value(rewards, strategy)
    return 1 if score > threshold else 0

def exact_match(pred, gold):
    """Check if prediction exactly matches gold answer"""
    return str(pred).strip().lower() == str(gold).strip().lower()

def load_cot_mapping(data_path, domain_categories):
    """Load CoT mapping from original dataset using q_id and cot_id"""
    cot_mapping = {}
    answer_mapping = {}
    
    for category in domain_categories:
        test_file = os.path.join(data_path, f"{category}_dataset.json")
        
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                test_data = json.load(f)
            
            for entry in test_data:
                q_id = entry["q_id"]
                answer_mapping[q_id] = entry["answer"]
                
                for cot_id, cot in zip(entry['cot_ids'], entry["cots"]):
                    cot_mapping[(q_id, cot_id)] = cot
    
    print(f"Loaded CoT mapping for {len(cot_mapping)} CoTs from {len(answer_mapping)} questions")
    return cot_mapping, answer_mapping

def evaluate_verification_performance(input_dir, cot_mapping, answer_mapping, strategy, threshold=0.5, domain_categories=None):
    """Evaluate verification performance for individual CoTs using q_id and cot_id matching"""
    if domain_categories is None:
        domain_categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                           'history', 'economics', 'math', 'business', 'philosophy', 
                           'health', 'engineering', 'computer_science', 'other']
    
    cot_results = []
    
    for category in domain_categories:
        eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
        
        if not os.path.exists(eval_file):
            continue
            
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        
        for entry in eval_data:
            q_id = entry["q_id"]
            gold_answer = answer_mapping.get(q_id)
            
            if gold_answer is None:
                continue
            
            cot_ids = entry.get("cot_ids", [])
            rewards = entry.get("rewards", [])
            parsed_answers = entry.get("parsed_answers", [])
            
            for i, (cot_id, reward, answer) in enumerate(zip(cot_ids, rewards, parsed_answers)):
                cot_key = (q_id, cot_id)
                if cot_key not in cot_mapping:
                    continue
                
                cot = cot_mapping[cot_key]
                cot_length = len(cot)
                
                if isinstance(reward, list) and len(reward) == 0:
                    continue
                
                reward_score = get_reward_value(reward, strategy)
                verification_pred = get_verification_prediction(reward, strategy, threshold)
                true_label = 1 if exact_match(answer, gold_answer) else 0
                
                cot_results.append({
                    'cot_length': cot_length,
                    'verification_pred': verification_pred,
                    'verification_score': reward_score,
                    'true_label': true_label,
                })
    
    return cot_results

def create_equal_count_bins(cot_results, num_bins):
    """Create exactly num_bins with equal counts"""
    lengths = [cot['cot_length'] for cot in cot_results]
    sorted_lengths = sorted(lengths)
    n_total = len(sorted_lengths)
    
    items_per_bin = n_total // num_bins
    remainder = n_total % num_bins
    
    bin_edges = []
    current_index = 0
    
    for i in range(num_bins):
        # Add one extra item to the first 'remainder' bins
        bin_size = items_per_bin + (1 if i < remainder else 0)
        
        if i == 0:
            bin_edges.append(sorted_lengths[0])
        
        current_index += bin_size
        
        if current_index >= n_total:
            bin_edges.append(sorted_lengths[-1] + 0.01)
            break
        else:
            # Set boundary just above current value to ensure proper binning
            boundary = sorted_lengths[current_index - 1] + 0.01
            bin_edges.append(boundary)
    
    # Ensure we have exactly num_bins + 1 edges
    if len(bin_edges) != num_bins + 1:
        bin_edges.append(sorted_lengths[-1] + 0.01)
    
    return np.array(bin_edges)

def compute_verification_performance_by_length(cot_results, bin_edges):
    """Compute verification performance metrics for each length bin"""
    bin_performance = []
    
    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        bin_cots = [cot for cot in cot_results if bin_min <= cot['cot_length'] < bin_max]
        
        if bin_cots:
            predicted_labels = [cot['verification_pred'] for cot in bin_cots]
            true_labels = [cot['true_label'] for cot in bin_cots]
            scores = [cot['verification_score'] for cot in bin_cots]
            
            f1, auprc = compute_verification_metrics(predicted_labels, true_labels, scores)
            
            bin_performance.append({
                'bin_range': (bin_min, bin_max),
                'f1_score': f1 * 100,  # Convert to percentage
                'auprc': auprc * 100,  # Convert to percentage
                'count': len(bin_cots)
            })
    
    return bin_performance

def main():
    parser = argparse.ArgumentParser(description="Analyze verification performance across CoT lengths")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to original dataset with CoTs")
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                       help="Model directories with evaluation results")
    parser.add_argument("--model_names", type=str, nargs='+', 
                       default=["DisPRM", "DisORM", "GenPRM", "GenORM"],
                       help="Names for the models")
    parser.add_argument("--strategies", type=str, nargs='+',
                        default=["min", "last", "mean", "mean"],
                       help="Strategies for each model")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for verification prediction")
    parser.add_argument("--num_bins", type=int, default=9,
                       help="Number of bins for CoT length analysis")
    
    args = parser.parse_args()

    # Validation
    if len(args.input_dirs) != len(args.model_names):
        if len(args.model_names) == 4:
            args.model_names = args.model_names[:len(args.input_dirs)]
        else:
            raise ValueError("Number of input_dirs must match number of model_names")
    
    if len(args.input_dirs) != len(args.strategies):
        raise ValueError("Number of input_dirs must match number of strategies")
    
    print(f"Analyzing {len(args.input_dirs)} models: {', '.join(args.model_names)}")
    print(f"Using {args.num_bins} equal-count bins")
    
    # Load CoT mapping
    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    cot_mapping, answer_mapping = load_cot_mapping(args.data_path, categories)
    
    if not cot_mapping:
        print("Error: No CoT mapping found.")
        return
    
    # Evaluate each model
    all_model_cot_results = []
    
    for i, (input_dir, strategy) in enumerate(zip(args.input_dirs, args.strategies)):
        cot_results = evaluate_verification_performance(
            input_dir, cot_mapping, answer_mapping, strategy, args.threshold, categories
        )
        all_model_cot_results.append(cot_results)
        print(f"{args.model_names[i]}: {len(cot_results)} CoTs")
    
    if not any(all_model_cot_results):
        print("Error: No valid results found.")
        return
    
    # Use model with most data for binning
    best_model_idx = max(range(len(all_model_cot_results)), key=lambda i: len(all_model_cot_results[i]))
    bin_edges = create_equal_count_bins(all_model_cot_results[best_model_idx], args.num_bins)
    
    # Compute performance for each model
    all_model_performance = []
    for model_idx in range(len(args.input_dirs)):
        if all_model_cot_results[model_idx]:
            performance = compute_verification_performance_by_length(all_model_cot_results[model_idx], bin_edges)
        else:
            performance = []
        all_model_performance.append(performance)
    
    # Print F1 results
    print(f"\nF1 Score by CoT Length ({args.num_bins} equal-count bins):")
    print("-" * (30 + len(args.model_names) * 12))
    
    header = f"{'Bin Range':<20} {'Count':<10}"
    for name in args.model_names:
        header += f"{name:<12}"
    print(header)
    print("-" * (30 + len(args.model_names) * 12))
    
    for i in range(args.num_bins):
        if i < len(all_model_performance[best_model_idx]):
            bin_info = all_model_performance[best_model_idx][i]
            bin_min, bin_max = bin_info['bin_range']
            count = bin_info['count']
            range_str = f"[{int(bin_min)},{int(bin_max)})"
            
            row = f"{range_str:<20} {count:<10}"
            
            for model_idx in range(len(args.model_names)):
                if i < len(all_model_performance[model_idx]):
                    f1 = all_model_performance[model_idx][i]['f1_score']
                    row += f"{f1:<12.2f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
    
    # Print AUPRC results
    print(f"\nAUPRC by CoT Length ({args.num_bins} equal-count bins):")
    print("-" * (30 + len(args.model_names) * 12))
    
    header = f"{'Bin Range':<20} {'Count':<10}"
    for name in args.model_names:
        header += f"{name:<12}"
    print(header)
    print("-" * (30 + len(args.model_names) * 12))
    
    for i in range(args.num_bins):
        if i < len(all_model_performance[best_model_idx]):
            bin_info = all_model_performance[best_model_idx][i]
            bin_min, bin_max = bin_info['bin_range']
            count = bin_info['count']
            range_str = f"[{int(bin_min)},{int(bin_max)})"
            
            row = f"{range_str:<20} {count:<10}"
            
            for model_idx in range(len(args.model_names)):
                if i < len(all_model_performance[model_idx]):
                    auprc = all_model_performance[model_idx][i]['auprc']
                    row += f"{auprc:<12.2f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

if __name__ == "__main__":
    main()