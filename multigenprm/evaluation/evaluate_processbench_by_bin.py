import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from datasets import load_dataset
from collections import defaultdict

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

def create_quantile_bins(cot_lengths, num_bins):
    """Create quantile-based bins"""
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(cot_lengths, quantiles)
    
    # Ensure integer edges and handle duplicates
    bin_edges = np.round(bin_edges).astype(int)
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    
    # If we have fewer unique edges than expected, pad the last edge
    if len(bin_edges) < num_bins + 1:
        bin_edges = np.append(bin_edges, max(cot_lengths) + 1)
    
    return bin_edges

def assign_to_quantile_bin(length, bin_edges):
    """Assign a length to appropriate quantile bin"""
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= length < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2

def evaluate_methods_by_length(datasets_data, eval_datasets, strategy="mean", num_rewards=None, threshold=0.5, num_bins=5):
    """Return predictions and ground truth labels binned by CoT length across all datasets"""
    # Collect all CoT lengths and valid entries across all datasets
    all_cot_lengths = []
    all_valid_entries = []
    
    for dataset_name, dataset in datasets_data.items():
        if dataset_name not in eval_datasets:
            continue
            
        mapping = {d["id"]: d for d in dataset}
        eval_dataset = eval_datasets[dataset_name]
        
        for entry in eval_dataset:
            if entry["id"] in mapping and "steps" in mapping[entry["id"]]:
                cot_length = len(mapping[entry["id"]]["steps"])
                all_cot_lengths.append(cot_length)
                all_valid_entries.append((entry, mapping[entry["id"]], cot_length))
    
    if not all_cot_lengths:
        print("Warning: No valid entries with 'steps' found!")
        return {}
    
    # Create quantile-based bins
    bin_edges = create_quantile_bins(all_cot_lengths, num_bins)
    actual_num_bins = len(bin_edges) - 1
    
    # Initialize bins
    bins = {}
    for i in range(actual_num_bins):
        bins[i] = {
            'y_true': [],
            'y_pred': [],
            'range': (bin_edges[i], bin_edges[i + 1] - 1),
            'lengths': []
        }
    
    # Assign entries to bins
    for entry, mapped_entry, cot_length in all_valid_entries:
        # Ground truth
        final_answer_correct = mapped_entry["final_answer_correct"]
        label = mapped_entry["label"]
        y = 1 if final_answer_correct else 0
        
        # Prediction
        r = get_reward_value(entry["rewards"], strategy, num_rewards)
        y_hat = 1 if r > threshold else 0
        
        # Assign to quantile bin
        bin_idx = assign_to_quantile_bin(cot_length, bin_edges)
        bins[bin_idx]['y_true'].append(y)
        bins[bin_idx]['y_pred'].append(y_hat)
        bins[bin_idx]['lengths'].append(cot_length)
    
    return bins

def calculate_metrics(y_true, y_pred):
    """Calculate metrics from predictions"""
    if len(y_true) == 0:
        return None
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'f1_score': f1,
        'total_samples': len(y_true)
    }

def process_single_method(input_dir, strategy, datasets_data, seeds, num_rewards=None, threshold=0.5, num_bins=5):
    """Process a single method (input directory + strategy combination)"""
    datasets = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    
    # Store F1 scores for each bin across seeds
    bin_f1_scores = {}
    bin_ranges = {}
    bin_sample_counts = {}
    
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
        
        # Get bins across all datasets
        bins = evaluate_methods_by_length(
            datasets_data, eval_datasets, 
            strategy, num_rewards, threshold, num_bins
        )
        
        # Calculate metrics for each bin
        for bin_idx, bin_data in bins.items():
            metrics = calculate_metrics(bin_data['y_true'], bin_data['y_pred'])
            if metrics:
                if bin_idx not in bin_f1_scores:
                    bin_f1_scores[bin_idx] = []
                    bin_ranges[bin_idx] = bin_data['range']
                    bin_sample_counts[bin_idx] = []
                
                bin_f1_scores[bin_idx].append(metrics['f1_score'] * 100)
                bin_sample_counts[bin_idx].append(metrics['total_samples'])
    
    # Calculate mean and std for each bin
    results = {}
    for bin_idx in bin_f1_scores.keys():
        f1_scores = bin_f1_scores[bin_idx]
        results[bin_idx] = {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0,
            'avg_samples': np.mean(bin_sample_counts[bin_idx]),
            'range': bin_ranges[bin_idx]
        }
    
    return results

def create_comparison_table(all_method_results, method_names, decimals=2):
    """Create a comparison table with methods as rows and bins as columns"""
    # Get all unique bin indices
    all_bin_indices = set()
    for results in all_method_results.values():
        all_bin_indices.update(results.keys())
    all_bin_indices = sorted(all_bin_indices)
    
    # Get bin ranges and sample counts (should be consistent across methods)
    bin_ranges = {}
    bin_sample_counts = {}
    for results in all_method_results.values():
        for bin_idx in results:
            if bin_idx not in bin_ranges:
                bin_ranges[bin_idx] = results[bin_idx]['range']
                bin_sample_counts[bin_idx] = results[bin_idx]['avg_samples']
    
    # Create table data
    table_data = []
    
    for method_name in method_names:
        if method_name not in all_method_results:
            continue
            
        row = {'Method': method_name}
        results = all_method_results[method_name]
        
        for bin_idx in all_bin_indices:
            if bin_idx in results:
                f1_mean = results[bin_idx]['f1_mean']
                f1_std = results[bin_idx]['f1_std']
                row[f'Bin_{bin_idx}'] = f"{f1_mean:.{decimals}f}±{f1_std:.{decimals}f}"
            else:
                row[f'Bin_{bin_idx}'] = "N/A"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    return df, bin_ranges, bin_sample_counts

def print_comparison_table(df, bin_ranges, bin_sample_counts, num_bins):
    """Print a nicely formatted comparison table"""
    print("\n" + "="*100)
    print("PERFORMANCE BY CoT LENGTH QUANTILE BINS - METHOD COMPARISON")
    print("="*100)
    
    # Print bin ranges and sample counts
    print("\nBin Ranges (CoT Length) and Sample Counts:")
    for i in range(num_bins):
        if i in bin_ranges:
            range_start, range_end = bin_ranges[i]
            sample_count = int(bin_sample_counts[i]) if i in bin_sample_counts else "N/A"
            print(f"  Bin {i}: {range_start}-{range_end} (n={sample_count})")
    
    print(f"\nF1 Scores (Mean ± Std):")
    print("-" * 100)
    
    # Create header
    header = f"{'Method':<25}"
    for i in range(num_bins):
        header += f"{'Bin ' + str(i):<15}"
    print(header)
    print("-" * 100)
    
    # Print each method's results
    for _, row in df.iterrows():
        line = f"{row['Method']:<25}"
        for i in range(num_bins):
            col_name = f'Bin_{i}'
            if col_name in row:
                value = row[col_name] if row[col_name] != "N/A" else "N/A"
                line += f"{value:<15}"
            else:
                line += f"{'N/A':<15}"
        print(line)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple methods by CoT length using quantile bins")
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                        help="List of input directories (one per method)")
    parser.add_argument("--strategies", type=str, nargs='+', required=True,
                        choices=['min', 'max', 'mean', 'last', 'prod'],
                        help="List of strategies (one per method, corresponds to input_dirs)")
    parser.add_argument("--method_names", type=str, nargs='+', default=None,
                        help="Custom names for methods (optional, defaults to dir_strategy)")
    parser.add_argument("--num_rewards", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_bins", type=int, default=5)
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
    
    print(f"Comparing methods by CoT length with {args.num_bins} quantile-based bins")
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
    all_method_results = {}
    
    for i, (input_dir, strategy, method_name) in enumerate(zip(args.input_dirs, args.strategies, method_names)):
        print(f"\nProcessing method {i+1}/{len(method_names)}: {method_name}")
        
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue
        
        results = process_single_method(
            input_dir, strategy, datasets_data, args.seeds,
            args.num_rewards, args.threshold, args.num_bins
        )
        
        if results:
            all_method_results[method_name] = results
            print(f"  ✓ Successfully processed {method_name}")
        else:
            print(f"  ✗ No valid results found for {method_name}")
    
    if not all_method_results:
        print("\nNo valid results found for any method!")
        return
    
    # Create and display comparison table
    df, bin_ranges, bin_sample_counts = create_comparison_table(all_method_results, method_names, args.decimals)
    print_comparison_table(df, bin_ranges, bin_sample_counts, args.num_bins)
    
    # Save to CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Methods successfully processed: {len(all_method_results)}")
    print(f"Total seeds: {len(args.seeds)}")
    print(f"Quantile bins: {args.num_bins}")
    
    # Find best performing method per bin
    print(f"\nBest performing method per bin:")
    for i in range(args.num_bins):
        col_name = f'Bin_{i}'
        if col_name in df.columns:
            # Extract F1 means for comparison
            bin_scores = {}
            for _, row in df.iterrows():
                if row[col_name] != "N/A":
                    # Extract mean from "mean±std" format
                    f1_mean = float(row[col_name].split('±')[0])
                    bin_scores[row['Method']] = f1_mean
            
            if bin_scores:
                best_method = max(bin_scores.items(), key=lambda x: x[1])
                range_info = f" ({bin_ranges[i][0]}-{bin_ranges[i][1]})" if i in bin_ranges else ""
                print(f"  Bin {i}{range_info}: {best_method[0]} ({best_method[1]:.{args.decimals}f})")
    
    # Print sample counts for plotting
    print(f"\nSample counts for plotting:")
    counts_for_plot = []
    for i in range(args.num_bins):
        if i in bin_sample_counts:
            count = int(bin_sample_counts[i])
            counts_for_plot.append(f"n={count}")
        else:
            counts_for_plot.append("n=???")
    print(f"counts = {counts_for_plot}")

if __name__ == "__main__":
    main()