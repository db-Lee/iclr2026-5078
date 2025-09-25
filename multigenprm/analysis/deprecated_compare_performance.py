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
    else:  # strategy == "last"
        return rewards[-1]

def load_and_prepare_data(input_dir, data_path, category):
    """Load data and create lookup tables"""
    eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
    data_file = os.path.join(data_path, f"{category}_dataset.json")
    
    if not os.path.exists(eval_file) or not os.path.exists(data_file):
        return None, None, None
    
    with open(eval_file) as f:
        eval_dataset = json.load(f)
    with open(data_file) as f:
        original_data = json.load(f)
    
    # Create lookup: q_id -> cot_ids from original data
    original_cot_mapping = {entry['q_id']: entry['cot_ids'] for entry in original_data}
    
    # Create lookup: (q_id, cot_id) -> (reward, answer, cot_length) from eval data and original data
    eval_lookup = {}
    cot_length_mapping = {}
    
    # Get CoT lengths from original data
    for entry in original_data:
        q_id = entry['q_id']
        for cot_id, cot in zip(entry['cot_ids'], entry['cots']):
            cot_length_mapping[(q_id, cot_id)] = len(cot)
    
    # Get rewards and answers from eval data
    for eval_entry in eval_dataset:
        q_id = eval_entry['q_id']
        for i, cot_id in enumerate(eval_entry['cot_ids']):
            cot_length = cot_length_mapping.get((q_id, cot_id), 0)
            eval_lookup[(q_id, cot_id)] = (eval_entry['rewards'][i], eval_entry['parsed_answers'][i], cot_length)
    
    return eval_dataset, original_cot_mapping, eval_lookup

def subsample_deterministically(eval_dataset, original_cot_mapping, eval_lookup, N_max, seed):
    """Create subsampled dataset deterministically"""
    subsampled = []
    
    for eval_entry in eval_dataset:
        q_id = eval_entry['q_id']
        original_cot_ids = original_cot_mapping[q_id]
        
        # Deterministic subsampling
        if len(original_cot_ids) <= N_max:
            selected_cot_ids = original_cot_ids
        else:
            # Deterministic seed per question
            deterministic_seed = int(hashlib.md5(f"{seed}_{q_id}".encode()).hexdigest()[:8], 16)
            np.random.seed(deterministic_seed)
            indices = np.random.choice(len(original_cot_ids), N_max, replace=False)
            selected_cot_ids = [original_cot_ids[i] for i in indices]
        
        # Get data for selected COTs
        rewards = []
        answers = []
        lengths = []
        for cot_id in selected_cot_ids:
            reward, answer, length = eval_lookup[(q_id, cot_id)]
            rewards.append(reward)
            answers.append(answer)
            lengths.append(length)
        
        # Calculate average length for this question
        avg_length = np.mean(lengths) if lengths else 0
        
        subsampled.append({
            'q_id': q_id,
            'answer': eval_entry['answer'],
            'rewards': rewards,
            'parsed_answers': answers,
            'avg_length': avg_length
        })
    
    return subsampled

def evaluate_methods(dataset, strategy="mean", include_reward_free=True):
    """Evaluate all methods including reward-free ones"""
    results = {}
    
    # Initialize results based on what we're computing
    if include_reward_free:
        results.update({'majority_vote': 0, 'pass_at_n': 0})
    results.update({'best_of_n': 0, 'weighted_vote': 0, 'oracle': 0})
    
    for entry in dataset:
        answers = entry['parsed_answers']
        rewards = entry['rewards']
        gold = entry['answer']

        # Reward-free methods (computed once)
        if include_reward_free:
            # Majority vote
            answer_counts = Counter(ans.strip().lower() for ans in answers)
            majority_pred = answer_counts.most_common(1)[0][0]
            if exact_match(majority_pred, gold):
                results['majority_vote'] += 1

            # Pass@N_max (if any answer is correct)
            if any(exact_match(ans, gold) for ans in answers):
                results['pass_at_n'] += 1

        # Reward-dependent methods
        # Best-of-N
        best_idx = max(range(len(answers)), key=lambda i: get_reward_value(rewards[i], strategy))
        if exact_match(answers[best_idx], gold):
            results['best_of_n'] += 1

        # Weighted vote
        vote_weights = defaultdict(float)
        for ans, r in zip(answers, rewards):
            vote_weights[ans.strip().lower()] += get_reward_value(r, strategy)
        weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
        if exact_match(weighted_pred, gold):
            results['weighted_vote'] += 1

        # Oracle (if any answer is correct) - same as pass@n but kept for consistency
        if any(exact_match(ans, gold) for ans in answers):
            results['oracle'] += 1

    return results

def create_equal_count_bins(datasets, num_bins):
    """Create exactly num_bins with equal counts"""
    # Collect all average lengths
    lengths = []
    for dataset in datasets:
        for entry in dataset:
            if entry['avg_length'] > 0:
                lengths.append(entry['avg_length'])
    
    if not lengths:
        return np.array([0, 1])
    
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

def evaluate_by_length_bins(datasets, bin_edges, strategy="mean", include_reward_free=True):
    """Evaluate performance for each length bin across multiple datasets"""
    bin_results = []
    
    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        
        # Collect all questions in this bin across all datasets
        bin_questions = []
        for dataset in datasets:
            for entry in dataset:
                if bin_min <= entry['avg_length'] < bin_max:
                    bin_questions.append(entry)
        
        if bin_questions:
            results = evaluate_methods(bin_questions, strategy, include_reward_free)
            total = len(bin_questions)
            
            result_dict = {
                'bin_range': (bin_min, bin_max),
                'count': total,
                'bin_center': (bin_min + bin_max) / 2
            }
            
            # Add reward-free metrics if computed
            if include_reward_free:
                result_dict.update({
                    'majority_vote': (results['majority_vote'] / total) * 100,
                    'pass_at_n': (results['pass_at_n'] / total) * 100,
                })
            
            # Add reward-dependent metrics
            result_dict.update({
                'best_of_n': (results['best_of_n'] / total) * 100,
                'weighted_vote': (results['weighted_vote'] / total) * 100,
                'oracle': (results['oracle'] / total) * 100,
            })
            
            bin_results.append(result_dict)
        else:
            result_dict = {
                'bin_range': (bin_min, bin_max),
                'count': 0,
                'bin_center': (bin_min + bin_max) / 2
            }
            
            # Initialize all metrics to 0
            if include_reward_free:
                result_dict.update({'majority_vote': 0, 'pass_at_n': 0})
            result_dict.update({'best_of_n': 0, 'weighted_vote': 0, 'oracle': 0})
            
            bin_results.append(result_dict)
    
    return bin_results

def evaluate_category(input_dir, data_path, category, num_runs, N_max, strategy, base_seed):
    eval_dataset, original_cot_mapping, eval_lookup = load_and_prepare_data(input_dir, data_path, category)
    
    if eval_dataset is None:
        return []
    
    all_datasets = []
    for run_idx in range(num_runs):
        dataset = subsample_deterministically(eval_dataset, original_cot_mapping, eval_lookup, N_max, base_seed + run_idx)
        all_datasets.append(dataset)
    
    return all_datasets

def main():
    parser = argparse.ArgumentParser(description="Analyze voting performance by average CoT length per question across multiple models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                       help="Model directories with evaluation results")
    parser.add_argument("--model_names", type=str, nargs='+', 
                       default=["DisPRM", "DisORM", "GenPRM", "GenORM"],
                       help="Names for the models")
    parser.add_argument("--strategies", type=str, nargs='+',
                        default=["min", "last", "mean", "mean"],
                       help="Strategies for each model (same length as input_dirs)")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max", type=int, default=8)
    parser.add_argument("--num_bins", type=int, default=8)
    args = parser.parse_args()

    # Validation
    if len(args.input_dirs) != len(args.model_names):
        if len(args.model_names) == 4:
            args.model_names = args.model_names[:len(args.input_dirs)]
        else:
            raise ValueError("Number of input_dirs must match number of model_names")
    
    if len(args.input_dirs) != len(args.strategies):
        raise ValueError("Number of input_dirs must match number of strategies")
    
    valid_strategies = ['min', 'max', 'mean', 'last']
    for strategy in args.strategies:
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")

    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    print(f"Analyzing voting performance for {len(args.input_dirs)} models: {', '.join(args.model_names)}")
    print(f"Strategies: {', '.join([f'{name}({strategy})' for name, strategy in zip(args.model_names, args.strategies)])}")
    print(f"Settings: N_max={args.N_max}, num_runs={args.num_runs}, seed={args.seed}, num_bins={args.num_bins}")
    
    # Collect all datasets for each model
    all_model_datasets = []
    
    for model_idx, (input_dir, strategy) in enumerate(zip(args.input_dirs, args.strategies)):
        print(f"\nProcessing {args.model_names[model_idx]}...")
        model_datasets = []
        
        for category in tqdm(categories, desc=f"Loading {args.model_names[model_idx]}"):
            try:
                datasets = evaluate_category(input_dir, args.data_path, category, 
                                           args.num_runs, args.N_max, strategy, args.seed)
                model_datasets.extend(datasets)
            except Exception as e:
                continue  # Skip categories with missing files
        
        all_model_datasets.append(model_datasets)
        
        if model_datasets:
            print(f"{args.model_names[model_idx]}: {len(model_datasets)} datasets with {sum(len(d) for d in model_datasets)} total questions")
        else:
            print(f"{args.model_names[model_idx]}: No valid datasets found")
    
    if not any(all_model_datasets):
        print("No valid datasets found for any model!")
        return
    
    # Find the model with the most data for binning
    best_model_idx = max(range(len(all_model_datasets)), key=lambda i: len(all_model_datasets[i]))
    
    if not all_model_datasets[best_model_idx]:
        print("No valid datasets found for binning!")
        return
    
    print(f"\nUsing {args.model_names[best_model_idx]} for binning (has most data: {len(all_model_datasets[best_model_idx])} datasets)")
    
    # Create equal-count bins based on average lengths from the model with most data
    bin_edges = create_equal_count_bins(all_model_datasets[best_model_idx], args.num_bins)
    print(f"\nCreated {len(bin_edges)-1} equal-count bins with boundaries: {bin_edges}")
    
    # Compute reward-free metrics once using the model with most data
    print("\nComputing reward-free metrics (MV and Pass@N)...")
    reward_free_bin_results = evaluate_by_length_bins(
        all_model_datasets[best_model_idx], bin_edges, 
        strategy="mean",  # Strategy doesn't matter for reward-free metrics
        include_reward_free=True
    )
    
    # Evaluate reward-dependent performance by bins for each model
    all_model_bin_results = []
    for model_idx in range(len(args.input_dirs)):
        print(f"Computing reward-dependent metrics for {args.model_names[model_idx]}...")
        if all_model_datasets[model_idx]:
            bin_results = evaluate_by_length_bins(
                all_model_datasets[model_idx], bin_edges, args.strategies[model_idx],
                include_reward_free=False  # Don't recompute reward-free metrics
            )
        else:
            # Create empty performance for models with no results
            bin_results = [{
                'bin_range': (bin_edges[i], bin_edges[i+1]),
                'best_of_n': 0, 'weighted_vote': 0, 'oracle': 0, 'count': 0,
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2
            } for i in range(len(bin_edges)-1)]
        all_model_bin_results.append(bin_results)
    
    # Print clean results table
    print(f"\nPerformance by Average CoT Length per Question:")
    print("-" * (40 + len(args.model_names) * 12))
    header = f"{'Length Range':<15} {'Count (%)':<12} {'MV':<8} {'Pass@N':<8} "
    for name in args.model_names:
        header += f"{name:<12} "
    print(header)
    print("-" * (40 + len(args.model_names) * 12))
    
    total_questions = sum(r['count'] for r in reward_free_bin_results)
    
    for i in range(len(reward_free_bin_results)):
        bin_min, bin_max = reward_free_bin_results[i]['bin_range']
        range_str = f"[{int(bin_min)},{int(bin_max)})"
        
        count = reward_free_bin_results[i]['count']
        percentage = count / total_questions * 100 if total_questions > 0 else 0
        count_str = f"{count} ({percentage:.1f}%)"
        
        mv_score = reward_free_bin_results[i]['majority_vote']
        pass_n_score = reward_free_bin_results[i]['pass_at_n']
        
        row = f"{range_str:<15} {count_str:<12} {mv_score:<8.2f} {pass_n_score:<8.2f} "
        
        for model_idx in range(len(args.model_names)):
            best_of_n = all_model_bin_results[model_idx][i]['best_of_n']
            row += f"{best_of_n:<12.2f} "
        print(row)

if __name__ == "__main__":
    main()