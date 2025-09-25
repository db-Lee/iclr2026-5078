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

def load_and_prepare_data(input_dir, model_seed, data_path, category):
    """Load data and create lookup tables from seed-specific directory"""
    # Data is in input_dir/seed/eval_category_dataset.json
    eval_file = os.path.join(input_dir, str(model_seed), f"eval_{category}_dataset.json")
    data_file = os.path.join(data_path, f"{category}_dataset.json")
    
    if not os.path.exists(eval_file) or not os.path.exists(data_file):
        return None, None, None
    
    with open(eval_file) as f:
        eval_dataset = json.load(f)
    with open(data_file) as f:
        original_data = json.load(f)
    
    # Create lookup: q_id -> cot_ids from original data
    original_cot_mapping = {entry['q_id']: entry['cot_ids'] for entry in original_data}
    original_temp_mapping = {entry['q_id']: entry['parsed_answer_correctness'] for entry in original_data}
    
    # Create lookup: (q_id, cot_id) -> (reward, answer) from eval data
    eval_lookup = {}
    for eval_entry in eval_dataset:
        q_id = eval_entry['q_id']        
        for i, (cot_id, parsed_answer_correctness) in enumerate(zip(eval_entry['cot_ids'], original_temp_mapping[q_id])):
            eval_lookup[(q_id, cot_id)] = (eval_entry['rewards'][i], parsed_answer_correctness)
    
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
        parsed_answer_correctnesses = []
        for cot_id in selected_cot_ids:
            reward, parsed_answer_correctness = eval_lookup[(q_id, cot_id)]
            rewards.append(reward)
            parsed_answer_correctnesses.append(parsed_answer_correctness)
        
        subsampled.append({
            'q_id': q_id,
            'rewards': rewards,
            'parsed_answer_correctnesses': parsed_answer_correctnesses
        })
    
    return subsampled

def evaluate_methods(dataset, strategy="mean"):
    results = {'best_of_n': 0, 'oracle': 0}
    
    for entry in dataset:
        rewards = entry['rewards']
        parsed_answer_correctnesses = entry['parsed_answer_correctnesses']
        
        # Best-of-N
        best_idx = max(range(len(rewards)), key=lambda i: get_reward_value(rewards[i], strategy))
        if parsed_answer_correctnesses[best_idx] == "true":
            results['best_of_n'] += 1
            
        # Oracle - check if any answer is correct (Pass@N)
        if "true" in parsed_answer_correctnesses:
            results['oracle'] += 1

    return {k: (v / len(dataset)) * 100 for k, v in results.items()}

def evaluate_category_for_model(input_dir, model_seed, data_path, category, num_runs, N_max, strategy, base_seed):
    """Evaluate a single category for a single model seed"""
    eval_dataset, original_cot_mapping, eval_lookup = load_and_prepare_data(input_dir, model_seed, data_path, category)
    
    if eval_dataset is None:
        return None
    
    all_results = {method: [] for method in ['best_of_n', 'oracle']}
    
    for run_idx in range(num_runs):
        dataset = subsample_deterministically(eval_dataset, original_cot_mapping, eval_lookup, N_max, base_seed + run_idx)
        results = evaluate_methods(dataset, strategy)
        for method, acc in results.items():
            all_results[method].append(acc)
    
    return all_results

def evaluate_category_for_model_multi_seed(input_dir, model_seeds, data_path, category, num_runs, N_max, strategy, base_seed):
    """Evaluate a single category for a single model across multiple model seeds"""
    # Store results for each model seed
    seed_results = []
    
    for model_seed in model_seeds:
        category_results = evaluate_category_for_model(
            input_dir, model_seed, data_path, category, num_runs, N_max, strategy, base_seed
        )
        if category_results is not None:
            seed_results.append(category_results)
        else:
            print(f"Warning: Could not load data for model_seed {model_seed} in category {category}")
            continue
    
    if not seed_results:
        return None
    
    # Collect all values from all model seeds and runs
    final_results = {}
    for method in ['best_of_n', 'oracle']:
        all_values = []
        for seed_result in seed_results:
            all_values.extend(seed_result[method])  # Add all runs from this model seed
        final_results[method] = all_values  # Keep all individual values
    
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                       help="Base seed for subsampling runs")
    parser.add_argument("--model_seeds", type=int, nargs='+', default=[0, 1, 2],
                       help="Model seeds to evaluate (corresponds to subdirectories)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                       help="Model directories: DisORM DisPRM GenORM GenPRM")
    parser.add_argument("--model_names", type=str, nargs='+', 
                       default=["DisORM", "DisPRM", "GenORM", "GenPRM"],
                       help="Names for the models")
    parser.add_argument("--strategies", type=str, nargs='+',
                        default=["last", "min", "mean", "mean"],
                       help="Strategies for each model")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max_values", type=int, nargs='+', default=[2, 4, 8, 16, 32, 64],
                       help="List of N_max values to evaluate")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    # Validation
    if len(args.input_dirs) != len(args.model_names):
        raise ValueError("Number of input_dirs must match number of model_names")
    
    if len(args.input_dirs) != len(args.strategies):
        raise ValueError("Number of input_dirs must match number of strategies")

    categories = ['gsm8k', 'math']
    # categories = ['gsm8k']
    
    print(f"Models: {', '.join(args.model_names)}")
    print(f"Strategies: {', '.join([f'{name}({strategy})' for name, strategy in zip(args.model_names, args.strategies)])}")
    print(f"Model seeds: {args.model_seeds}")
    print(f"N_max values: {args.N_max_values}")
    print(f"Settings: num_runs={args.num_runs}, base_seed={args.seed}")
    print(f"Total evaluations per model: {len(args.model_seeds) * args.num_runs}")
    
    # For each N_max value, create a table
    for N_max in args.N_max_values:
        print(f"\n{'='*80}")
        print(f"Results for N_max = {N_max}")
        print(f"{'='*80}")
        
        # Collect results for all models and categories
        all_model_results = {}
        
        for model_idx, (input_dir, model_name, strategy) in enumerate(zip(args.input_dirs, args.model_names, args.strategies)):
            print(f"\nEvaluating {model_name} with strategy {strategy} across model seeds {args.model_seeds}...")
            model_results = {}
            
            for category in tqdm(categories, desc=f"Processing {model_name}"):
                category_results = evaluate_category_for_model_multi_seed(
                    input_dir, args.model_seeds, args.data_path, category, args.num_runs, N_max, strategy, args.seed
                )
                if category_results is not None:
                    model_results[category] = category_results
            
            all_model_results[model_name] = model_results
        
        # ==== BEST-OF-N TABLE ARRAY FORMAT ====
        print(f"\nBest-of-N Array for N_max = {N_max}")
        print("    [")
        
        # Model results (Best-of-N)
        for model_name in args.model_names:
            if model_name in all_model_results and all_model_results[model_name]:
                model_results = all_model_results[model_name]
                
                # Calculate overall average across categories and all model seeds/runs
                all_scores = []
                for cat in categories:
                    if cat in model_results:
                        # model_results[cat]['best_of_n'] contains all values from all model seeds and runs
                        all_scores.extend(model_results[cat]['best_of_n'])
                
                if all_scores:
                    overall = np.mean(all_scores)
                else:
                    overall = 0.0
                
                # Category results
                cat_results = []
                for cat in categories:
                    if cat in model_results:
                        # Average across all model seeds and runs for this category
                        scores = model_results[cat]['best_of_n']
                        result = np.mean(scores)
                    else:
                        result = 0.0  # N/A becomes 0.0
                    cat_results.append(result)
                
                # Add overall at the beginning
                cat_results.insert(0, overall)
                model_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in cat_results]) + f"],  # {model_name}"
                print(f"        {model_array_str}")
            else:
                # Model has no results - all zeros
                all_zeros = [0.0] * (len(categories) + 1)  # +1 for overall
                model_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in all_zeros]) + f"],  # {model_name}"
                print(f"        {model_array_str}")
        
        # ==== ORACLE (Pass@N) RESULTS ====
        # Calculate oracle results across all models for this N_max
        all_oracle_scores = []
        for model_name in args.model_names:
            if model_name in all_model_results and all_model_results[model_name]:
                model_results = all_model_results[model_name]
                for cat in categories:
                    if cat in model_results:
                        all_oracle_scores.extend(model_results[cat]['oracle'])
        
        if all_oracle_scores:
            oracle_overall = np.mean(all_oracle_scores)
            
            # Category-specific oracle results (averaged across all models)
            oracle_cat_results = []
            for cat in categories:
                cat_oracle_scores = []
                for model_name in args.model_names:
                    if model_name in all_model_results and all_model_results[model_name]:
                        model_results = all_model_results[model_name]
                        if cat in model_results:
                            cat_oracle_scores.extend(model_results[cat]['oracle'])
                
                if cat_oracle_scores:
                    oracle_cat_results.append(np.mean(cat_oracle_scores))
                else:
                    oracle_cat_results.append(0.0)
            
            # Add overall at the beginning
            oracle_cat_results.insert(0, oracle_overall)
            oracle_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in oracle_cat_results]) + f"],  # Pass@{N_max}"
            print(f"        {oracle_array_str}")
        else:
            # No oracle results - all zeros
            all_zeros = [0.0] * (len(categories) + 1)  # +1 for overall
            oracle_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in all_zeros]) + f"],  # Pass@{N_max}"
            print(f"        {oracle_array_str}")
        
        print("    ],")
        

if __name__ == "__main__":
    main()