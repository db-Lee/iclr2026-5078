import os
import json
import argparse
import numpy as np
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm

def exact_match(pred, gold):
    return str(pred).strip().lower() == str(gold).strip().lower()

def get_reward_value(rewards, strategy):
    rewards = np.array(rewards)
    rewards[np.isnan(rewards)] = 0.5
    if strategy == "min":
        return np.min(rewards).item()
    elif strategy == "max":
        return np.max(rewards).item()
    elif strategy == "mean":
        return np.mean(rewards).item()
    elif strategy == "prod":
        return np.prod(rewards).item()
    else:  # strategy == "last"
        return rewards[-1]

def load_and_prepare_data(input_dir, data_path, category):
    """Load data and create lookup tables"""
    eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
    data_file = os.path.join(data_path, f"{category}_dataset.json")
    
    if not os.path.exists(eval_file) or not os.path.exists(data_file):
        return None, None, None, None
    
    with open(eval_file) as f:
        eval_dataset = json.load(f)
    with open(data_file) as f:
        original_data = json.load(f)
    
    # Create lookup: q_id -> cot_ids from original data
    original_cot_mapping = {entry['q_id']: entry['cot_ids'] for entry in original_data}
    
    # Create lookup: (q_id, cot_id) -> cot_length from original data
    cot_length_mapping = {}
    for entry in original_data:
        for cot_id, cot in zip(entry["cot_ids"], entry["cots"]):
            cot_length_mapping[(entry["q_id"], cot_id)] = len(cot)
    
    # Create lookup: (q_id, cot_id) -> (reward, answer) from eval data
    eval_lookup = {}
    for eval_entry in eval_dataset:
        q_id = eval_entry['q_id']
        for i, cot_id in enumerate(eval_entry['cot_ids']):
            eval_lookup[(q_id, cot_id)] = (eval_entry['rewards'][i], eval_entry['parsed_answers'][i])
    
    return eval_dataset, original_cot_mapping, eval_lookup, cot_length_mapping

def evaluate_all_data(input_dirs, model_names, strategies, data_path, categories, 
                     num_runs, N_max, seed, num_bins):
    """Evaluate all models and collect sample data for length analysis"""
    
    all_samples = []  # Store all samples with their results and lengths
    
    for model_idx, (input_dir, model_name, strategy) in enumerate(zip(input_dirs, model_names, strategies)):
        print(f"Processing {model_name}...")
        
        for category in tqdm(categories, desc=f"Categories for {model_name}"):
            eval_dataset, original_cot_mapping, eval_lookup, cot_length_mapping = load_and_prepare_data(
                input_dir, data_path, category)
            
            if eval_dataset is None:
                continue
            
            for run_idx in range(num_runs):
                # Sample data for this run
                for eval_entry in eval_dataset:
                    q_id = eval_entry['q_id']
                    original_cot_ids = original_cot_mapping[q_id]
                    
                    # Deterministic subsampling
                    if len(original_cot_ids) <= N_max:
                        selected_cot_ids = original_cot_ids
                    else:
                        deterministic_seed = int(hashlib.md5(f"{seed + run_idx}_{q_id}".encode()).hexdigest()[:8], 16)
                        np.random.seed(deterministic_seed)
                        indices = np.random.choice(len(original_cot_ids), N_max, replace=False)
                        selected_cot_ids = [original_cot_ids[i] for i in indices]
                    
                    # Get data for selected COTs
                    rewards = []
                    answers = []
                    cot_lengths = []
                    for cot_id in selected_cot_ids:
                        if (q_id, cot_id) in eval_lookup:
                            reward, answer = eval_lookup[(q_id, cot_id)]
                            cot_length = cot_length_mapping.get((q_id, cot_id), 0)
                            rewards.append(reward)
                            answers.append(answer)
                            cot_lengths.append(cot_length)
                    
                    if not rewards:
                        continue
                        
                    # Calculate average length for this sample
                    avg_length = np.mean(cot_lengths)
                    gold = eval_entry['answer']
                    
                    # Filter valid answers
                    valid_answers = [(ans, r) for ans, r in zip(answers, rewards) if ans and str(ans).strip()]
                    if not valid_answers:
                        continue
                        
                    valid_answer_texts = [ans for ans, _ in valid_answers]
                    valid_rewards = [r for _, r in valid_answers]
                    
                    # Evaluate methods
                    results = {}
                    
                    # Majority vote
                    answer_counts = Counter(ans.strip().lower() for ans in valid_answer_texts)
                    if answer_counts:
                        majority_pred = answer_counts.most_common(1)[0][0]
                        results['majority_vote'] = exact_match(majority_pred, gold)
                    
                    # Best-of-N
                    if model_name != 'majority_vote':  # Only for actual models
                        best_idx = max(range(len(valid_answer_texts)), 
                                     key=lambda i: get_reward_value(valid_rewards[i], strategy))
                        results[model_name] = exact_match(valid_answer_texts[best_idx], gold)
                    
                    # Weighted vote
                    vote_weights = defaultdict(float)
                    for ans, r in zip(valid_answer_texts, valid_rewards):
                        vote_weights[ans.strip().lower()] += get_reward_value(r, strategy)
                    if vote_weights:
                        weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
                        results[f'{model_name}_weighted'] = exact_match(weighted_pred, gold)
                    
                    # Oracle
                    results['oracle'] = any(exact_match(ans, gold) for ans in answers if ans and str(ans).strip())
                    
                    # Store sample
                    sample = {
                        'avg_length': avg_length,
                        'results': results,
                        'model': model_name
                    }
                    all_samples.append(sample)
    
    return all_samples

def analyze_length_performance(all_samples, num_bins, model_names):
    """Analyze performance by length bins"""
    
    if not all_samples:
        print("No samples to analyze")
        return
    
    # Create equal-width bins based on length range
    lengths = [s['avg_length'] for s in all_samples]
    min_length = min(lengths)
    max_length = max(lengths)
    bin_edges = np.linspace(min_length, max_length, num_bins + 1)
    
    print(f"\nLength-wise Performance Analysis (Equal-width bins)")
    print(f"Length range: {min_length:.1f} - {max_length:.1f} words")
    print(f"Bin edges (words): {[f'{edge:.1f}' for edge in bin_edges]}")
    print("="*80)
    
    # Create header
    methods = ['majority_vote'] + model_names + [f'{m}_weighted' for m in model_names] + ['oracle']
    header = f"{'Bin':<8} {'Range':<20} {'Count':<8}"
    for method in methods:
        header += f" {method:<12}"
    print(header)
    print("-" * len(header))
    
    # Analyze each bin
    for bin_idx in range(num_bins):
        min_len = bin_edges[bin_idx]
        max_len = bin_edges[bin_idx + 1]
        
        # Filter samples in this bin
        bin_samples = []
        for sample in all_samples:
            length = sample['avg_length']
            # Use proper bin assignment for equal-width bins
            if bin_idx == num_bins - 1:  # Last bin includes max value
                if min_len <= length <= max_len:
                    bin_samples.append(sample)
            else:
                if min_len <= length < max_len:
                    bin_samples.append(sample)
        
        count = len(bin_samples)
        if count == 0:
            row = f"{bin_idx:<8} [{min_len:.1f}, {max_len:.1f}]      {count:<8}"
            for _ in methods:
                row += f" {'N/A':<12}"
            print(row)
            continue
        
        # Calculate accuracy for each method
        row = f"{bin_idx:<8} [{min_len:.1f}, {max_len:.1f}]      {count:<8}"
        
        for method in methods:
            if method == 'majority_vote' or method == 'oracle':
                # These are consistent across models
                correct = sum(1 for s in bin_samples if s['results'].get(method, False))
            else:
                # Model-specific methods
                model_samples = [s for s in bin_samples if s['model'] in method or method.startswith(s['model'])]
                if not model_samples:
                    row += f" {'N/A':<12}"
                    continue
                correct = sum(1 for s in model_samples if s['results'].get(method, False))
                count = len(model_samples)
            
            accuracy = (correct / count) * 100 if count > 0 else 0
            row += f" {accuracy:<12.1f}"
        
        print(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--model_names", type=str, nargs='+', required=True)
    parser.add_argument("--strategies", type=str, nargs='+', required=True)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max", type=int, default=32)
    parser.add_argument("--num_bins", type=int, default=4)
    args = parser.parse_args()

    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    print(f"Models: {', '.join(args.model_names)}")
    print(f"Settings: N_max={args.N_max}, num_runs={args.num_runs}, num_bins={args.num_bins}")
    
    # Collect all sample data
    all_samples = evaluate_all_data(
        args.input_dirs, args.model_names, args.strategies, args.data_path, 
        categories, args.num_runs, args.N_max, args.seed, args.num_bins
    )
    
    # Analyze length-wise performance
    analyze_length_performance(all_samples, args.num_bins, args.model_names)

if __name__ == "__main__":
    main()