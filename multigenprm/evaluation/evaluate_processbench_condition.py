import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

def calculate_dataset_performance(datasets_data, eval_datasets, strategy="mean", num_rewards=None, threshold=0.5):
    """Calculate performance on good examples (label != -1 and final_answer_correct) per dataset"""
    dataset_performance = {}
    
    for dataset_name, dataset in datasets_data.items():
        if dataset_name not in eval_datasets:
            continue
            
        mapping = {d["id"]: d for d in dataset}
        eval_dataset = eval_datasets[dataset_name]
        
        # Filter for good examples: label != -1 and final_answer_correct
        good_examples = []
        for entry in eval_dataset:
            if entry["id"] in mapping:
                mapped_entry = mapping[entry["id"]]
                if mapped_entry["label"] != -1 and mapped_entry["final_answer_correct"]:
                    good_examples.append((entry, mapped_entry))
        
        if not good_examples:
            dataset_performance[dataset_name] = {
                'total_samples': 0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
            continue
        
        # Calculate predictions for good examples
        y_true = []
        y_pred = []
        
        for entry, mapped_entry in good_examples:
            # Ground truth (always 1 for good examples since final_answer_correct is True)
            y_true.append(1)
            
            # Prediction based on reward threshold
            r = get_reward_value(entry["rewards"], strategy, num_rewards)
            y_hat = 1 if r > threshold else 0
            y_pred.append(y_hat)
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred, zero_division=0) * 100
        recall = recall_score(y_true, y_pred, zero_division=0) * 100
        
        dataset_performance[dataset_name] = {
            'total_samples': len(good_examples),
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    return dataset_performance

def process_single_method(input_dir, strategy, datasets_data, seeds, num_rewards=None, threshold=0.5):
    """Process a single method for dataset performance analysis"""
    datasets = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    
    # Store dataset performance across seeds
    dataset_performances = {dataset: {'f1_scores': [], 'accuracies': [], 'precisions': [], 'recalls': [], 'sample_counts': []} 
                          for dataset in datasets}
    
    for seed in seeds:
        # Load all evaluation data for this seed
        eval_datasets = {}
        for dataset_name in datasets:
            output_file = os.path.join(input_dir, seed, f"eval_{dataset_name}_dataset.json")
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    eval_datasets[dataset_name] = json.load(f)
        
        if not eval_datasets:
            continue
        
        # Calculate dataset performance for good examples
        dataset_perf = calculate_dataset_performance(
            datasets_data, eval_datasets, strategy, num_rewards, threshold
        )
        
        for dataset_name, perf in dataset_perf.items():
            if dataset_name in dataset_performances:
                dataset_performances[dataset_name]['f1_scores'].append(perf['f1_score'])
                dataset_performances[dataset_name]['accuracies'].append(perf['accuracy'])
                dataset_performances[dataset_name]['precisions'].append(perf['precision'])
                dataset_performances[dataset_name]['recalls'].append(perf['recall'])
                dataset_performances[dataset_name]['sample_counts'].append(perf['total_samples'])
    
    # Calculate mean and std for dataset performance
    dataset_results = {}
    for dataset_name, perfs in dataset_performances.items():
        if perfs['f1_scores']:
            dataset_results[dataset_name] = {
                'f1_mean': np.mean(perfs['f1_scores']),
                'f1_std': np.std(perfs['f1_scores'], ddof=1) if len(perfs['f1_scores']) > 1 else 0.0,
                'accuracy_mean': np.mean(perfs['accuracies']),
                'accuracy_std': np.std(perfs['accuracies'], ddof=1) if len(perfs['accuracies']) > 1 else 0.0,
                'precision_mean': np.mean(perfs['precisions']),
                'precision_std': np.std(perfs['precisions'], ddof=1) if len(perfs['precisions']) > 1 else 0.0,
                'recall_mean': np.mean(perfs['recalls']),
                'recall_std': np.std(perfs['recalls'], ddof=1) if len(perfs['recalls']) > 1 else 0.0,
                'avg_samples': np.mean(perfs['sample_counts'])
            }
        else:
            dataset_results[dataset_name] = {
                'f1_mean': 0.0, 'f1_std': 0.0, 
                'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                'precision_mean': 0.0, 'precision_std': 0.0,
                'recall_mean': 0.0, 'recall_std': 0.0,
                'avg_samples': 0
            }
    
    return dataset_results

def print_performance_table(all_dataset_results, method_names, datasets, decimals=2):
    """Print performance table for good examples (label != -1 and final_answer_correct) by dataset"""
    print("\n" + "="*140)
    print("PERFORMANCE ON GOOD EXAMPLES (label != -1 AND final_answer_correct) BY DATASET")
    print("="*140)
    
    metrics = ['F1', 'Accuracy', 'Precision', 'Recall']
    metric_keys = ['f1', 'accuracy', 'precision', 'recall']
    
    for metric, metric_key in zip(metrics, metric_keys):
        print(f"\n{metric} Scores (Mean ± Std):")
        print("-" * 120)
        
        # Create header
        header = f"{'Method':<20}"
        for dataset in datasets:
            header += f"{dataset.upper():<18}"
        header += f"{'OVERALL':<18}"
        print(header)
        print("-" * 120)
        
        for method_name in method_names:
            if method_name not in all_dataset_results:
                continue
                
            line = f"{method_name:<20}"
            dataset_results = all_dataset_results[method_name]
            
            # Individual datasets
            dataset_scores = []
            
            for dataset in datasets:
                if dataset in dataset_results and dataset_results[dataset]['avg_samples'] > 0:
                    mean = dataset_results[dataset][f'{metric_key}_mean']
                    std = dataset_results[dataset][f'{metric_key}_std']
                    line += f"{mean:.{decimals}f}±{std:.{decimals}f}    "
                    dataset_scores.append(mean)
                else:
                    line += f"{'N/A':<18}"
            
            # Calculate overall performance as mean across datasets
            if dataset_scores:
                overall_mean = np.mean(dataset_scores)
                overall_std = np.std(dataset_scores, ddof=1) if len(dataset_scores) > 1 else 0.0
                line += f"{overall_mean:.{decimals}f}±{overall_std:.{decimals}f}"
            else:
                line += "N/A"
                
            print(line)
    
    # Print sample counts
    print(f"\nSample Counts per Dataset:")
    print("-" * 120)
    header = f"{'Method':<20}"
    for dataset in datasets:
        header += f"{dataset.upper():<18}"
    header += f"{'TOTAL':<18}"
    print(header)
    print("-" * 120)
    
    for method_name in method_names:
        if method_name not in all_dataset_results:
            continue
            
        line = f"{method_name:<20}"
        dataset_results = all_dataset_results[method_name]
        total_samples = 0
        
        for dataset in datasets:
            if dataset in dataset_results:
                samples = int(dataset_results[dataset]['avg_samples'])
                line += f"{samples:<18}"
                total_samples += samples
            else:
                line += f"{'0':<18}"
        
        line += f"{total_samples:<18}"
        print(line)

def main():
    parser = argparse.ArgumentParser(description="Analyze method performance on good examples by dataset")
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                        help="List of input directories (one per method)")
    parser.add_argument("--strategies", type=str, nargs='+', required=True,
                        choices=['min', 'max', 'mean', 'last', 'prod'],
                        help="List of strategies (one per method, corresponds to input_dirs)")
    parser.add_argument("--method_names", type=str, nargs='+', default=None,
                        help="Custom names for methods (optional, defaults to dir_strategy)")
    parser.add_argument("--num_rewards", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--seeds", type=str, nargs='+', default=['0', '1', '2'])
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save results table to CSV file")

    args = parser.parse_args()

    # Validate input arguments
    if len(args.input_dirs) != len(args.strategies):
        print("Error: Number of input_dirs must match number of strategies!")
        return

    datasets = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    
    print(f"Analyzing performance on good examples (label != -1 and final_answer_correct)")
    print(f"Number of methods: {len(args.input_dirs)}")
    print(f"Threshold: {args.threshold}, Seeds: {args.seeds}")
    
    # Create method names
    if args.method_names:
        if len(args.method_names) != len(args.input_dirs):
            print("Error: Number of method_names must match number of input_dirs!")
            return
        method_names = args.method_names
    else:
        method_names = []
        for i, (dir, strategy) in enumerate(zip(args.input_dirs, args.strategies)):
            base_name = f"{os.path.basename(dir)}_{strategy}"
            # Handle duplicate names by adding index
            if base_name in method_names:
                base_name = f"{base_name}_{i}"
            method_names.append(base_name)
    
    print("Methods to compare:")
    for i, (input_dir, strategy, method_name) in enumerate(zip(args.input_dirs, args.strategies, method_names)):
        print(f"  {i+1}. {method_name}: {input_dir} | {strategy}")
    
    # Load datasets once
    print("\nLoading datasets...")
    datasets_data = {}
    for dataset_name in datasets:
        datasets_data[dataset_name] = load_dataset("Qwen/ProcessBench", split=dataset_name)
    
    # Process each method
    all_dataset_results = {}
    
    for i, (input_dir, strategy, method_name) in enumerate(zip(args.input_dirs, args.strategies, method_names)):
        print(f"\nProcessing method {i+1}/{len(method_names)}: {method_name}")
        
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue
        
        dataset_results = process_single_method(
            input_dir, strategy, datasets_data, args.seeds,
            args.num_rewards, args.threshold
        )
        
        if dataset_results:
            all_dataset_results[method_name] = dataset_results
            print(f"  ✓ Successfully processed {method_name}")
        else:
            print(f"  ✗ No valid results found for {method_name}")
    
    if not all_dataset_results:
        print("\nNo valid results found for any method!")
        return
    
    # Print performance table
    print_performance_table(all_dataset_results, method_names, datasets, args.decimals)
    
    # Save to CSV if requested
    if args.output_csv:
        # Create a flattened DataFrame for CSV export
        csv_data = []
        for method_name, results in all_dataset_results.items():
            row = {'Method': method_name}
            for dataset in datasets:
                if dataset in results:
                    row[f'{dataset}_F1'] = f"{results[dataset]['f1_mean']:.{args.decimals}f}±{results[dataset]['f1_std']:.{args.decimals}f}"
                    row[f'{dataset}_Accuracy'] = f"{results[dataset]['accuracy_mean']:.{args.decimals}f}±{results[dataset]['accuracy_std']:.{args.decimals}f}"
                    row[f'{dataset}_Samples'] = int(results[dataset]['avg_samples'])
                else:
                    row[f'{dataset}_F1'] = "N/A"
                    row[f'{dataset}_Accuracy'] = "N/A" 
                    row[f'{dataset}_Samples'] = 0
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Methods successfully processed: {len(all_dataset_results)}")
    print(f"Total seeds: {len(args.seeds)}")
    print(f"Condition: label != -1 AND final_answer_correct == True")

if __name__ == "__main__":
    main()