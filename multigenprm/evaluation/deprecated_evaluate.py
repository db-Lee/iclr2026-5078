import os
import json
import argparse
import numpy as np
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

def evaluate_methods(dataset, N_max, distinct_questions=None, strategy="mean"):
    results = {'majority_vote': 0, 'best_of_n': 0, 'weighted_vote': 0, 'oracle': 0}
        
    for entry in dataset:        
        # Only shuffle if we have more answers than N_max
        if len(entry['parsed_answers']) <= N_max:
            indices = list(range(len(entry['parsed_answers'])))
        else:
            indices = np.random.permutation(len(entry['parsed_answers']))[:N_max]
        
        answers = [entry['parsed_answers'][i] for i in indices]
        rewards = [entry['rewards'][i] for i in indices]
        gold = entry['answer']

        # Majority vote
        answer_counts = Counter(ans.strip().lower() for ans in answers)
        majority_pred = answer_counts.most_common(1)[0][0]
        if exact_match(majority_pred, gold):
            results['majority_vote'] += 1

        # Best-of-N
        best_idx = max(range(len(answers)), 
                      key=lambda i: get_reward_value(rewards[i], strategy))
        if exact_match(answers[best_idx], gold):
            results['best_of_n'] += 1

        # Weighted vote
        vote_weights = defaultdict(float)
        for ans, r in zip(answers, rewards):
            vote_weights[ans.strip().lower()] += get_reward_value(r, strategy)
        weighted_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
        if exact_match(weighted_pred, gold):
            results['weighted_vote'] += 1

        # Oracle
        if any(exact_match(ans, gold) for ans in answers):
            results['oracle'] += 1

    # Denominator is number of distinct questions
    distinct_questions = len(dataset) if distinct_questions is None else distinct_questions
    return {k: v / distinct_questions for k, v in results.items()}

def evaluate_single_category(input_dir, data_path, category, num_runs, N_max, strategy):
    """Evaluate a single category and return results"""
    # Load eval dataset
    eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
    if not os.path.exists(eval_file):
        print(f"Warning: {eval_file} not found, skipping {category}")
        return None
        
    with open(eval_file) as f:
        dataset = json.load(f)
    
    # Get distinct questions count
    distinct_questions = None
    if data_path:
        data_file = os.path.join(data_path, f"{category}_dataset.json")
        if os.path.exists(data_file):
            with open(data_file) as f:
                distinct_questions = len(set(entry['q_id'] for entry in json.load(f)))
    
    # Run evaluations
    all_results = {method: [] for method in ['majority_vote', 'best_of_n', 'weighted_vote', 'oracle']}
    
    for _ in range(num_runs):
        results = evaluate_methods(dataset, N_max, distinct_questions, strategy)
        for method, acc in results.items():
            all_results[method].append(acc * 100)
    
    return all_results, distinct_questions

def print_category_results(category, all_results, distinct_questions, decimals=0):
    """Print results for a single category in compact table format"""
    print(f"\nRESULTS FOR {category.upper()} (mean±std):")
    if distinct_questions:
        print(f"Distinct questions: {distinct_questions}")
    
    print(f"{'Method':<6} {'Result':>10}")
    print("-" * 17)
    
    method_names = {'majority_vote': 'MV', 'best_of_n': 'BoN', 'weighted_vote': 'WMV', 'oracle': 'Pass@N'}
    
    for method_key, method_label in method_names.items():
        arr = np.array(all_results[method_key])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        result_str = f"{mean_val:.{decimals}f}±{std_val:.{decimals}f}"
        print(f"{method_label:<6} {result_str:>10}")
    
    majority_mean = np.mean(all_results['majority_vote'])
    bon_mean = np.mean(all_results['best_of_n'])
    print(f"\nBoN improvement: {((bon_mean / majority_mean - 1) * 100):+.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate voting methods with relative performance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="./local_datasets/test")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                              'history', 'economics', 'math', 'business', 'philosophy', 
                              'health', 'engineering', 'computer_science', 'other', 'all'])
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max", type=int, default=16)
    parser.add_argument("--strategy", type=str, default="mean",
                       choices=['min', 'max', 'mean', 'last'])
    parser.add_argument("--decimals", type=int, default=2,
                       help="Number of decimal places to display in results (default: 0)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    if args.category == "all":
        categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                      'history', 'economics', 'math', 'business', 'philosophy', 
                      'health', 'engineering', 'computer_science', 'other']
        
        print("Evaluating all categories...")
        print(f"Settings: N_max={args.N_max}, strategy={args.strategy}, num_runs={args.num_runs}")
        
        # Store results for summary
        category_results = {}
        
        for category in tqdm(categories, desc="Processing categories"):
            result = evaluate_single_category(
                args.input_dir, args.data_path, category, 
                args.num_runs, args.N_max, args.strategy
            )
            
            if result is not None:
                all_results, distinct_questions = result
                category_results[category] = (all_results, distinct_questions)        
        
        valid_categories = [cat for cat in categories if cat in category_results]
        
        # Calculate column width based on decimals
        # Format: "XX.XX±X.XX" -> need enough space for the numbers
        if args.decimals == 0:
            col_width = 6  # "75±2"
        elif args.decimals == 1:
            col_width = 9  # "75.2±2.1"
        else:  # args.decimals >= 2
            col_width = 11  # "75.23±2.14"
        
        # Calculate statistics across categories for each method
        # For each run, compute mean across categories, then mean/std across runs
        method_means = {}
        for method_key in ['majority_vote', 'best_of_n', 'weighted_vote', 'oracle']:
            # Get results for all categories and all runs
            all_category_results = []
            for cat in valid_categories:
                all_results, _ = category_results[cat]
                all_category_results.append(all_results[method_key])  # List of results across runs
            
            # For each run, compute mean across categories
            run_means = []
            for run_idx in range(args.num_runs):
                # Mean across categories for this specific run
                run_mean = np.mean([cat_results[run_idx] for cat_results in all_category_results])
                run_means.append(run_mean)
            
            # Mean and std of run_means
            method_means[method_key] = {
                'mean_of_means': np.mean(run_means),
                'std_of_means': np.std(run_means)
            }
        
        # Print header with Mean first, then category names
        cat_short = [cat[:4] for cat in valid_categories]  # First 4 chars
        header_cats = " ".join(f"{cat:<{col_width}}" for cat in cat_short)
        header_line = f"{'Method':<6} {'Mean':<{col_width}} {header_cats}"
        print(header_line)
        print("-" * len(header_line))
        
        # Print MV first
        method_key = 'majority_vote'
        method_label = 'MV'
        row_results = []
        for cat in valid_categories:
            all_results, _ = category_results[cat]
            mean_val = np.mean(all_results[method_key])
            std_val = np.std(all_results[method_key])
            formatted_result = f"{mean_val:.{args.decimals}f}±{std_val:.{args.decimals}f}"
            row_results.append(f"{formatted_result:<{col_width}}")
        
        # Add mean column first
        mean_of_means = method_means[method_key]['mean_of_means']
        std_of_means = method_means[method_key]['std_of_means']
        mean_formatted = f"{mean_of_means:.{args.decimals}f}±{std_of_means:.{args.decimals}f}"
        
        results_row = " ".join(row_results)
        print(f"{method_label:<6} {mean_formatted:<{col_width}} {results_row}")
        
        # Separator
        print("-" * len(header_line))
        
        # Print BoN and WMV
        for method_key, method_label in [('best_of_n', 'BoN'), ('weighted_vote', 'WMV')]:
            row_results = []
            for cat in valid_categories:
                all_results, _ = category_results[cat]
                mean_val = np.mean(all_results[method_key])
                std_val = np.std(all_results[method_key])
                formatted_result = f"{mean_val:.{args.decimals}f}±{std_val:.{args.decimals}f}"
                row_results.append(f"{formatted_result:<{col_width}}")
            
            # Add mean column first
            mean_of_means = method_means[method_key]['mean_of_means']
            std_of_means = method_means[method_key]['std_of_means']
            mean_formatted = f"{mean_of_means:.{args.decimals}f}±{std_of_means:.{args.decimals}f}"
            
            results_row = " ".join(row_results)
            print(f"{method_label:<6} {mean_formatted:<{col_width}} {results_row}")
        
        # Separator
        print("-" * len(header_line))
        
        # Print relative improvements (BoN and WMV relative to P@K)
        for method_key, method_label in [('best_of_n', 'BoN'), ('weighted_vote', 'WMV')]:
            row_results = []
            for cat in valid_categories:
                all_results, _ = category_results[cat]
                oracle_results, _ = category_results[cat]
                
                # Calculate ratio for each run, then mean/std
                ratios = []
                for run_idx in range(args.num_runs):
                    oracle_val = oracle_results['oracle'][run_idx]
                    method_val = all_results[method_key][run_idx]
                    if oracle_val > 0:  # Avoid division by zero
                        ratio = (method_val / oracle_val) * 100
                    else:
                        ratio = 0
                    ratios.append(ratio)
                
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                formatted_result = f"{mean_ratio:.{args.decimals}f}±{std_ratio:.{args.decimals}f}"
                row_results.append(f"{formatted_result:<{col_width}}")
            
            # Calculate mean ratio across categories: mean(method/oracle) for each run
            all_run_ratios = []
            for run_idx in range(args.num_runs):
                run_ratios = []
                for cat in valid_categories:
                    all_results, _ = category_results[cat]
                    oracle_results, _ = category_results[cat]
                    oracle_val = oracle_results['oracle'][run_idx]
                    method_val = all_results[method_key][run_idx]
                    if oracle_val > 0:
                        ratio = (method_val / oracle_val) * 100
                    else:
                        ratio = 0
                    run_ratios.append(ratio)
                # Mean ratio across categories for this run
                all_run_ratios.append(np.mean(run_ratios))
            
            overall_mean_ratio = np.mean(all_run_ratios)
            overall_std_ratio = np.std(all_run_ratios)
            mean_formatted = f"{overall_mean_ratio:.{args.decimals}f}±{overall_std_ratio:.{args.decimals}f}"
            
            rel_label = f"{method_label}/P"
            results_row = " ".join(row_results)
            print(f"{rel_label:<6} {mean_formatted:<{col_width}} {results_row}")
        
        # Separator
        print("-" * len(header_line))
        
        # Print Pass@N
        method_key = 'oracle'
        method_label = f'P@{args.N_max}'
        row_results = []
        for cat in valid_categories:
            all_results, _ = category_results[cat]
            mean_val = np.mean(all_results[method_key])
            std_val = np.std(all_results[method_key])
            formatted_result = f"{mean_val:.{args.decimals}f}±{std_val:.{args.decimals}f}"
            row_results.append(f"{formatted_result:<{col_width}}")
        
        # Add mean column first
        mean_of_means = method_means[method_key]['mean_of_means']
        std_of_means = method_means[method_key]['std_of_means']
        mean_formatted = f"{mean_of_means:.{args.decimals}f}±{std_of_means:.{args.decimals}f}"
        
        results_row = " ".join(row_results)
        print(f"{method_label:<6} {mean_formatted:<{col_width}} {results_row}")
        
    else:
        # Single category evaluation
        result = evaluate_single_category(
            args.input_dir, args.data_path, args.category, 
            args.num_runs, args.N_max, args.strategy
        )
        
        if result is not None:
            all_results, distinct_questions = result
            print_category_results(args.category, all_results, distinct_questions, args.decimals)

if __name__ == "__main__":
    main()