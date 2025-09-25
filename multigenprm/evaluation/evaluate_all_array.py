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
        return None, None, None
    
    with open(eval_file) as f:
        eval_dataset = json.load(f)
    with open(data_file) as f:
        original_data = json.load(f)
    
    # Create lookup: q_id -> cot_ids from original data
    original_cot_mapping = {entry['q_id']: entry['cot_ids'] for entry in original_data}
    
    # Create lookup: (q_id, cot_id) -> (reward, answer) from eval data
    eval_lookup = {}
    for eval_entry in eval_dataset:
        q_id = eval_entry['q_id']
        for i, cot_id in enumerate(eval_entry['cot_ids']):
            eval_lookup[(q_id, cot_id)] = (eval_entry['rewards'][i], eval_entry['parsed_answers'][i])
    
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
        for cot_id in selected_cot_ids:
            reward, answer = eval_lookup[(q_id, cot_id)]
            rewards.append(reward)
            answers.append(answer)
        
        subsampled.append({
            'q_id': q_id,
            'answer': eval_entry['answer'],
            'rewards': rewards,
            'parsed_answers': answers
        })
    
    return subsampled

def evaluate_methods(dataset, strategy="mean"):
    results = {'majority_vote': 0, 'best_of_n': 0, 'weighted_vote': 0, 'oracle': 0}
    
    for entry in dataset:
        answers = entry['parsed_answers']
        rewards = entry['rewards']
        gold = entry['answer']

        # Filter out empty or None answers
        valid_answers = [(ans, r) for ans, r in zip(answers, rewards) if ans and str(ans).strip()]
        
        if not valid_answers:
            # If no valid answers, skip this entry (all methods get 0 for this question)
            continue
            
        valid_answer_texts = [ans for ans, _ in valid_answers]
        valid_rewards = [r for _, r in valid_answers]

        # Majority vote
        answer_counts = Counter(ans.strip().lower() for ans in valid_answer_texts)
        if answer_counts:  # Additional safety check
            majority_pred = answer_counts.most_common(1)[0][0]
            if exact_match(majority_pred, gold):
                results['majority_vote'] += 1

        # Best-of-N
        best_idx = max(range(len(valid_answer_texts)), key=lambda i: get_reward_value(valid_rewards[i], strategy))
        if exact_match(valid_answer_texts[best_idx], gold):
            results['best_of_n'] += 1

        # Weighted vote
        vote_weights = defaultdict(float)
        for ans, r in zip(valid_answer_texts, valid_rewards):
            vote_weights[ans.strip().lower()] += get_reward_value(r, strategy)
        if vote_weights:  # Additional safety check
            weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
            if exact_match(weighted_pred, gold):
                results['weighted_vote'] += 1

        # Oracle (check original answers list for any correct answer)
        if any(exact_match(ans, gold) for ans in answers if ans and str(ans).strip()):
            results['oracle'] += 1

    return {k: (v / len(dataset)) * 100 for k, v in results.items()}

def evaluate_category_for_model(input_dir, data_path, category, num_runs, N_max, strategy, base_seed):
    """Evaluate a single category for a single model"""
    eval_dataset, original_cot_mapping, eval_lookup = load_and_prepare_data(input_dir, data_path, category)
    
    if eval_dataset is None:
        return None
    
    all_results = {method: [] for method in ['majority_vote', 'best_of_n', 'weighted_vote', 'oracle']}
    
    for run_idx in range(num_runs):
        dataset = subsample_deterministically(eval_dataset, original_cot_mapping, eval_lookup, N_max, base_seed + run_idx)
        results = evaluate_methods(dataset, strategy)
        for method, acc in results.items():
            all_results[method].append(acc)
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
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

    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    print(f"Models: {', '.join(args.model_names)}")
    print(f"Strategies: {', '.join([f'{name}({strategy})' for name, strategy in zip(args.model_names, args.strategies)])}")
    print(f"N_max values: {args.N_max_values}")
    print(f"Settings: num_runs={args.num_runs}, seed={args.seed}")
    
    # For each N_max value, create a table
    for N_max in args.N_max_values:
        print(f"\n{'='*80}")
        print(f"Results for N_max = {N_max}")
        print(f"{'='*80}")
        
        # Collect results for all models and categories
        all_model_results = {}
        
        for model_idx, (input_dir, model_name, strategy) in enumerate(zip(args.input_dirs, args.model_names, args.strategies)):
            print(f"\nEvaluating {model_name} with strategy {strategy}...")
            model_results = {}
            
            for category in tqdm(categories, desc=f"Processing {model_name}"):
                try:
                    category_results = evaluate_category_for_model(
                        input_dir, args.data_path, category, args.num_runs, N_max, strategy, args.seed
                    )
                    if category_results is not None:
                        model_results[category] = category_results
                except Exception as e:
                    print(f"Error processing {model_name}/{category}: {e}")
                    continue
            
            all_model_results[model_name] = model_results
        
        # Find first available model for shared calculations
        first_model = None
        for model_name, model_results in all_model_results.items():
            if model_results:
                first_model = model_name
                break
        
        # ==== BEST-OF-N TABLE ARRAY FORMAT ====
        print(f"\nBest-of-N Array for N_max = {N_max}")
        print("    [")
        
        # Majority Vote
        if first_model:
            mv_results = []
            mv_overall_runs = []
            
            for run_idx in range(args.num_runs):
                category_scores = []
                for cat in categories:
                    if cat in all_model_results[first_model]:
                        score = all_model_results[first_model][cat]['majority_vote'][run_idx]
                        category_scores.append(score)
                if category_scores:
                    mv_overall_runs.append(np.mean(category_scores))
            
            mv_overall = np.mean(mv_overall_runs)
            
            for cat in categories:
                if cat in all_model_results[first_model]:
                    scores = all_model_results[first_model][cat]['majority_vote']
                    result = np.mean(scores)
                else:
                    result = 0.0  # N/A becomes 0.0
                mv_results.append(result)
            
            # Add overall at the beginning
            mv_results.insert(0, mv_overall)
            mv_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in mv_results]) + "],  # MV"
            print(f"        {mv_array_str}")
        
        # Model results (Best-of-N)
        for model_name in args.model_names:
            if model_name in all_model_results and all_model_results[model_name]:
                model_results = all_model_results[model_name]
                
                # Calculate overall average across categories
                overall_runs = []
                for run_idx in range(args.num_runs):
                    category_scores = []
                    for cat in categories:
                        if cat in model_results:
                            score = model_results[cat]['best_of_n'][run_idx]
                            category_scores.append(score)
                    if category_scores:
                        overall_runs.append(np.mean(category_scores))
                
                overall = np.mean(overall_runs)
                
                # Category results
                cat_results = []
                for cat in categories:
                    if cat in model_results:
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
        
        # Pass@N (Oracle)
        if first_model:
            oracle_results = []
            oracle_overall_runs = []
            
            for run_idx in range(args.num_runs):
                category_scores = []
                for cat in categories:
                    if cat in all_model_results[first_model]:
                        score = all_model_results[first_model][cat]['oracle'][run_idx]
                        category_scores.append(score)
                if category_scores:
                    oracle_overall_runs.append(np.mean(category_scores))
            
            oracle_overall = np.mean(oracle_overall_runs)
            
            for cat in categories:
                if cat in all_model_results[first_model]:
                    scores = all_model_results[first_model][cat]['oracle']
                    result = np.mean(scores)
                else:
                    result = 0.0  # N/A becomes 0.0
                oracle_results.append(result)
            
            # Add overall at the beginning
            oracle_results.insert(0, oracle_overall)
            oracle_array_str = "[" + ", ".join([f"{val:.{args.decimals}f}" for val in oracle_results]) + f"]   # Pass@{N_max}"
            print(f"        {oracle_array_str}")
        
        print("    ],")
        
        # ==== WEIGHTED MAJORITY VOTE TABLE ARRAY FORMAT ====
        print(f"\nWeighted Majority Vote Array for N_max = {N_max}")
        print("    [")
        
        # Majority Vote (same as above)
        if first_model:
            print(f"        {mv_array_str}")
        
        # Model results (Weighted Vote)
        for model_name in args.model_names:
            if model_name in all_model_results and all_model_results[model_name]:
                model_results = all_model_results[model_name]
                
                # Calculate overall average across categories
                overall_runs = []
                for run_idx in range(args.num_runs):
                    category_scores = []
                    for cat in categories:
                        if cat in model_results:
                            score = model_results[cat]['weighted_vote'][run_idx]
                            category_scores.append(score)
                    if category_scores:
                        overall_runs.append(np.mean(category_scores))
                
                overall = np.mean(overall_runs)
                
                # Category results
                cat_results = []
                for cat in categories:
                    if cat in model_results:
                        scores = model_results[cat]['weighted_vote']
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
        
        # Pass@N (Oracle) - same as above
        if first_model:
            print(f"        {oracle_array_str}")
        
        print("    ],")

if __name__ == "__main__":
    main()