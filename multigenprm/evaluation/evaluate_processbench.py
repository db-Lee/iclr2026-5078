import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from datasets import load_dataset

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

def evaluate_methods(dataset, eval_dataset, strategy="mean", num_rewards=None, threshold=0.5):
    """Return raw predictions and ground truth labels"""
    y_true = []
    y_pred = []
    
    mapping = {d["id"]: d for d in dataset}
    
    for entry in eval_dataset:
        # Ground truth
        final_answer_correct = mapping[entry["id"]]["final_answer_correct"]
        # label = mapping[entry["id"]]["label"]
        y = 1 if final_answer_correct else 0
        y_true.append(y)
                
        # Use rewards with specified strategy
        r = get_reward_value(entry["rewards"], strategy, num_rewards)
        y_hat = 1 if r > threshold else 0
        y_pred.append(y_hat)
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate all metrics from predictions"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_samples': len(y_true)
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate rewards for generated critiques")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="mean",
                        choices=['min', 'max', 'mean', 'last', 'prod'])
    parser.add_argument("--num_rewards", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for reward-based classification")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--seeds", type=str, nargs='+',
                        default=['0', '1', '2'])

    args = parser.parse_args()

    # Define all datasets to process
    datasets = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    
    print(f"Evaluating all datasets with strategy={args.strategy}, threshold={args.threshold}")
    print(f"Using seeds: {args.seeds}")
    
    # Load datasets once
    print("Loading datasets...")
    dataset_data = {}
    for dataset_name in datasets:
        dataset_data[dataset_name] = load_dataset("Qwen/ProcessBench", split=dataset_name)
        print(f"Loaded {dataset_name}")
    
    # Store results for each seed
    seed_results = {}
    
    for seed in args.seeds:
        print(f"\nProcessing seed {seed}...")
        dataset_predictions = {}
        all_y_true = []
        all_y_pred = []
        
        for dataset_name in datasets:
            output_file = os.path.join(args.input_dir, seed, f"eval_{dataset_name}_dataset.json")
            
            if os.path.exists(output_file):
                print(f"Loading data from: {output_file}")
                with open(output_file, 'r') as f:
                    eval_data = json.load(f)

                # Get predictions for this dataset using pre-loaded data
                y_true, y_pred = evaluate_methods(dataset_data[dataset_name], eval_data, args.strategy, args.num_rewards, args.threshold)
                dataset_predictions[dataset_name] = (y_true, y_pred)
                
                # Accumulate for overall metrics
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
            else:
                print(f"Warning: No file found for {dataset_name} in seed {seed}, skipping")
        
        if not dataset_predictions:
            print(f"No valid datasets found for seed {seed}!")
            continue
        
        # Calculate metrics for each dataset for this seed
        seed_dataset_results = {}
        for dataset_name, (y_true, y_pred) in dataset_predictions.items():
            seed_dataset_results[dataset_name] = calculate_metrics(y_true, y_pred)
        
        # Calculate overall metrics for this seed
        overall_results = calculate_metrics(all_y_true, all_y_pred)
        
        seed_results[seed] = {
            'overall': overall_results,
            'datasets': seed_dataset_results
        }
    
    if not seed_results:
        print("No valid seeds found!")
        return
    
    # Calculate statistics across seeds
    all_datasets = set()
    for seed_data in seed_results.values():
        all_datasets.update(seed_data['datasets'].keys())
    all_datasets = sorted(list(all_datasets))
    
    # Collect F1 scores across seeds for each dataset and overall
    overall_f1_scores = [seed_results[seed]['overall']['f1_score'] * 100 for seed in seed_results.keys()]
    dataset_f1_scores = {}
    
    for dataset_name in all_datasets:
        scores = []
        for seed in seed_results.keys():
            if dataset_name in seed_results[seed]['datasets']:
                scores.append(seed_results[seed]['datasets'][dataset_name]['f1_score'] * 100)
        dataset_f1_scores[dataset_name] = scores
    
    # Calculate mean and std (convert to regular Python floats)
    overall_f1_mean = float(np.mean(overall_f1_scores))
    overall_f1_std = float(np.std(overall_f1_scores, ddof=1)) if len(overall_f1_scores) > 1 else 0.0
    
    dataset_f1_means = {}
    dataset_f1_stds = {}
    for dataset_name in all_datasets:
        scores = dataset_f1_scores[dataset_name]
        dataset_f1_means[dataset_name] = float(np.mean(scores))
        dataset_f1_stds[dataset_name] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    
    # Determine column width (wider to accommodate mean±std format)
    col_width = max(12, 8 + 2 * args.decimals)
    
    # Print results table
    print(f"\n" + "="*60)
    print(f"RESULTS BY DATASET (Mean ± Std across {len(seed_results)} seeds)")
    print(f"="*60)
    
    dataset_short = [ds[:4] for ds in all_datasets]
    header_datasets = " ".join(f"{ds:<{col_width}}" for ds in dataset_short)
    header = f"{'Metric':<8} {'Overall':<{col_width}} {header_datasets}"
    print(header)
    print("-" * len(header))
    
    # F1 scores with mean ± std
    dataset_f1_results = []
    for dataset_name in all_datasets:
        mean_val = dataset_f1_means[dataset_name]
        std_val = dataset_f1_stds[dataset_name]
        f1_formatted = f"{mean_val:.{args.decimals}f}±{std_val:.{args.decimals}f}"[:col_width].ljust(col_width)
        dataset_f1_results.append(f1_formatted)
    
    overall_f1_formatted = f"{overall_f1_mean:.{args.decimals}f}±{overall_f1_std:.{args.decimals}f}"[:col_width].ljust(col_width)
    print(f"{'F1':<8} {overall_f1_formatted} {' '.join(dataset_f1_results)}")
    
    # Return means and stds as regular Python float lists
    means = [overall_f1_mean] + [dataset_f1_means[dataset_name] for dataset_name in all_datasets]
    stds = [overall_f1_std] + [dataset_f1_stds[dataset_name] for dataset_name in all_datasets]
    
    return means, stds
    
if __name__ == "__main__":
    main()