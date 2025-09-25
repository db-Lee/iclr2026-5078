import os
import json
import argparse
import numpy as np
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm

def exact_match(pred, gold):
    return str(pred).strip().lower() == str(gold).strip().lower()

def get_reward_value(rewards, strategy, num_rewards=None):
    rewards = np.array(rewards[:num_rewards])
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

def evaluate_methods(dataset, strategy="mean", num_rewards=-1):
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
        best_idx = max(range(len(valid_answer_texts)), key=lambda i: get_reward_value(valid_rewards[i], strategy, num_rewards))
        if exact_match(valid_answer_texts[best_idx], gold):
            results['best_of_n'] += 1

        # Weighted vote
        vote_weights = defaultdict(float)
        for ans, r in zip(valid_answer_texts, valid_rewards):
            vote_weights[ans.strip().lower()] += get_reward_value(r, strategy, num_rewards)
        if vote_weights:  # Additional safety check
            weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
            if exact_match(weighted_pred, gold):
                results['weighted_vote'] += 1

        # Oracle (check original answers list for any correct answer)
        if any(exact_match(ans, gold) for ans in answers if ans and str(ans).strip()):
            results['oracle'] += 1

    return {k: (v / len(dataset)) * 100 for k, v in results.items()}

def evaluate_category(input_dir, data_path, category, num_runs, N_max, strategy, num_rewards, base_seed):
    eval_dataset, original_cot_mapping, eval_lookup = load_and_prepare_data(input_dir, data_path, category)
    
    all_results = {method: [] for method in ['majority_vote', 'best_of_n', 'weighted_vote', 'oracle']}
    
    for run_idx in range(num_runs):
        dataset = subsample_deterministically(eval_dataset, original_cot_mapping, eval_lookup, N_max, base_seed + run_idx)
        results = evaluate_methods(dataset, strategy, num_rewards)
        for method, acc in results.items():
            all_results[method].append(acc)
    
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)    
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max", type=int, default=8)
    parser.add_argument("--strategy", type=str, default="mean", choices=['min', 'max', 'mean', 'last', 'prod'])
    parser.add_argument("--num_rewards", type=int, default=None)
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    print(f"Settings: N_max={args.N_max}, strategy={args.strategy}, num_runs={args.num_runs}, seed={args.seed}")
    
    category_results = {}
    for category in tqdm(categories):
        all_results = evaluate_category(args.input_dir, args.data_path, category, 
                                       args.num_runs, args.N_max, args.strategy, args.num_rewards, args.seed)
        category_results[category] = all_results
    
    valid_categories = list(category_results.keys())
    
    # Calculate column width
    col_width = 11 if args.decimals >= 2 else (9 if args.decimals == 1 else 6)
    
    # Calculate overall averages
    method_stats = {}
    for method in ['majority_vote', 'best_of_n', 'weighted_vote', 'oracle']:
        run_averages = []
        for run_idx in range(args.num_runs):
            total_score = sum(category_results[cat][method][run_idx] for cat in valid_categories)
            avg_score = total_score / len(valid_categories)
            run_averages.append(avg_score)
        
        method_stats[method] = {
            'mean': np.mean(run_averages),
            'std': np.std(run_averages)
        }
    
    # Print results
    cat_headers = " ".join(f"{cat[:4]:<{col_width}}" for cat in categories)
    header = f"{'Method':<6} {'Mean':<{col_width}} {cat_headers}"
    print(header)
    print("-" * len(header))
    
    for method, label in [('majority_vote', 'MV'), ('best_of_n', 'BoN'), ('weighted_vote', 'WMV'), ('oracle', f'P@{args.N_max}')]:
        # Category results
        cat_results = []
        for cat in categories:
            if cat in category_results:
                mean_val = np.mean(category_results[cat][method])
                std_val = np.std(category_results[cat][method])
                result = f"{mean_val:.{args.decimals}f}±{std_val:.{args.decimals}f}"
            else:
                result = "N/A"
            cat_results.append(f"{result:<{col_width}}")
        
        # Overall mean
        overall = f"{method_stats[method]['mean']:.{args.decimals}f}±{method_stats[method]['std']:.{args.decimals}f}"
        
        print(f"{label:<6} {overall:<{col_width}} {' '.join(cat_results)}")

if __name__ == "__main__":
    main()