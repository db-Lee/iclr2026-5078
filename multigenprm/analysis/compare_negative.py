import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def count_negative_rewards(rewards):
    """Count the number of negative rewards in a list"""
    if isinstance(rewards, list):
        return sum(1 for r in rewards if r < 0.5)
    else:
        return 1 if rewards < 0.5 else 0

def get_total_rewards_count(rewards):
    """Get total number of rewards"""
    if isinstance(rewards, list):
        return len(rewards)
    else:
        return 1

def evaluate_negative_proportion_with_cot_map(dataset, cot_length_map):
    """Evaluate negative reward proportion using external CoT length mapping"""
    question_results = []
    
    for entry in dataset:        
        # Skip entries with no answers or rewards
        if not entry.get('parsed_answers') or not entry.get('rewards'):
            continue
        
        # Get CoT length from mapping
        q_id = entry.get('q_id')
        cot_length = cot_length_map.get(q_id, 0) if q_id is not None else 0

        # Calculate negative reward proportion for all answers
        total_negative = 0
        total_rewards = 0
        
        for reward_list in entry['rewards']:
            if reward_list:  # Skip empty reward lists
                total_negative += count_negative_rewards(reward_list)
                total_rewards += get_total_rewards_count(reward_list)
        
        # Calculate proportion
        negative_proportion = total_negative / total_rewards if total_rewards > 0 else 0
        
        question_results.append({
            'q_id': q_id,
            'cot_length': cot_length,
            'negative_proportion': negative_proportion,
            'total_negative': total_negative,
            'total_rewards': total_rewards,
            'category': entry.get('category', 'unknown')
        })

    return question_results

def load_all_categories_data(input_dir):
    """Load data from all categories and combine"""
    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    all_data = []
    for category in categories:
        eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                category_data = json.load(f)
                # Add category info to each entry
                for entry in category_data:
                    entry['category'] = category
                all_data.extend(category_data)
    
    return all_data

def create_cot_length_map(dataset):
    """Create a mapping from q_id to CoT length"""
    cot_length_map = {}
    for entry in dataset:
        q_id = entry.get('q_id')
        if q_id is not None and 'cots' in entry:
            cot_length_map[q_id] = sum([len(cot) for cot in entry['cots']]) / len(entry['cots'])
    return cot_length_map

def create_length_bins(question_results, num_bins=10):
    """Create bins with equal sample sizes using quantiles"""
    lengths = [q['cot_length'] for q in question_results]
    if not lengths:
        return np.array([0, 1])
    
    # Use quantiles to create equal-sample bins
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(lengths, quantiles)
    bin_edges[-1] += 1  # Ensure max value is included
    
    return bin_edges

def compute_negative_proportion_by_length(question_results, bin_edges):
    """Compute negative reward proportion for each length bin"""
    bin_performance = []
    
    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        bin_questions = [q for q in question_results if bin_min <= q['cot_length'] < bin_max]
        
        if bin_questions:
            # Calculate average negative proportion for this bin
            avg_negative_proportion = np.mean([q['negative_proportion'] for q in bin_questions])
            total_negatives = sum(q['total_negative'] for q in bin_questions)
            total_rewards = sum(q['total_rewards'] for q in bin_questions)
            overall_proportion = total_negatives / total_rewards if total_rewards > 0 else 0
            
            bin_performance.append({
                'bin_range': (bin_min, bin_max),
                'avg_negative_proportion': avg_negative_proportion * 100,  # Convert to percentage
                'overall_negative_proportion': overall_proportion * 100,  # Convert to percentage
                'count': len(bin_questions),
                'total_negative': total_negatives,
                'total_rewards': total_rewards,
                'bin_center': (bin_min + bin_max) / 2
            })
        else:
            bin_performance.append({
                'bin_range': (bin_min, bin_max),
                'avg_negative_proportion': 0,
                'overall_negative_proportion': 0,
                'count': 0,
                'total_negative': 0,
                'total_rewards': 0,
                'bin_center': (bin_min + bin_max) / 2
            })
    
    return bin_performance

def plot_negative_proportion_comparison(model1_performance, model2_performance, model1_name, model2_name, 
                                      overall_prop1, overall_prop2, num_runs, output_path=None):
    """Create comparison histogram plot for negative reward proportions"""
    
    # Extract data for plotting
    bin_centers = [bp['bin_center'] for bp in model1_performance]
    model1_prop = [bp['overall_negative_proportion'] for bp in model1_performance]
    model2_prop = [bp['overall_negative_proportion'] for bp in model2_performance]
    counts = [bp['count'] for bp in model1_performance]  # Assuming same binning
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Main comparison plot
    x = np.arange(len(bin_centers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, model1_prop, width, label=f'{model1_name} (Overall: {overall_prop1:.1f}%)',
                    alpha=0.7, color='lightblue')
    bars2 = ax1.bar(x + width/2, model2_prop, width, label=f'{model2_name} (Overall: {overall_prop2:.1f}%)',
                    alpha=0.7, color='lightcoral')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if model1_prop[i] > 0:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                    f'{model1_prop[i]:.1f}%', ha='center', va='bottom', fontsize=8)
        if model2_prop[i] > 0:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                    f'{model2_prop[i]:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('CoT Length Bins (Equal Sample Size)')
    ax1.set_ylabel('Negative Reward Proportion (%)')
    ax1.set_title(f'Negative Reward Proportion Comparison by CoT Length\n(Averaged over {num_runs} runs)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(model1_performance[i]["bin_range"][0])}-{int(model1_performance[i]["bin_range"][1])}' 
                        for i in range(len(model1_performance))], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Question count histogram
    ax2.bar(x, counts, alpha=0.6, color='gray')
    ax2.set_xlabel('CoT Length Bins (Equal Sample Size)')
    ax2.set_ylabel('Question Count')
    ax2.set_title('Number of Questions per Length Bin')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(model1_performance[i]["bin_range"][0])}-{int(model1_performance[i]["bin_range"][1])}' 
                        for i in range(len(model1_performance))], rotation=45, ha='right')
    
    # Add count labels
    for i, count in enumerate(counts):
        if count > 0:
            ax2.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare negative reward proportions between two models")
    parser.add_argument("--data_path", type=str, required=True, help="Data path for CoT length mapping")
    parser.add_argument("--input_dir1", type=str, required=True, help="First model directory")
    parser.add_argument("--input_dir2", type=str, required=True, help="Second model directory")
    parser.add_argument("--model1_name", type=str, default="Model 1", help="Name for first model")
    parser.add_argument("--model2_name", type=str, default="Model 2", help="Name for second model")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of evaluation runs for averaging")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for CoT length histogram")
    parser.add_argument("--output_plot", type=str, default=None, help="Path to save the plot")
    
    args = parser.parse_args()

    print(f"Comparing negative reward proportions: {args.model1_name} vs {args.model2_name}")
    print(f"Settings: num_runs={args.num_runs}, num_bins={args.num_bins}")
    
    # Load data from both models
    print("Loading data...")
    model1_data = load_all_categories_data(args.input_dir1)
    model2_data = load_all_categories_data(args.input_dir2)
    
    print(f"Model 1: {len(model1_data)} questions")
    print(f"Model 2: {len(model2_data)} questions")
    
    # Create CoT length mapping
    print("Creating CoT length mapping...")
    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                  'history', 'economics', 'math', 'business', 'philosophy', 
                  'health', 'engineering', 'computer_science', 'other']
    
    all_data = []
    for category in categories:
        file = os.path.join(args.data_path, f"{category}_dataset.json")
        if os.path.exists(file):
            with open(file) as f:
                category_data = json.load(f)
                # Add category info to each entry
                for entry in category_data:
                    entry['category'] = category
                all_data.extend(category_data)        
    cot_length_map = create_cot_length_map(all_data)
    print(f"Found CoT information for {len(cot_length_map)} questions")
    
    # Run multiple evaluations and average results
    print("Running evaluations...")
    
    model1_results = []
    model2_results = []
    model1_overall_props = []
    model2_overall_props = []
    
    for run in tqdm(range(args.num_runs), desc="Evaluation runs"):
        # Evaluate negative proportions for both models
        results1 = evaluate_negative_proportion_with_cot_map(model1_data, cot_length_map)
        results2 = evaluate_negative_proportion_with_cot_map(model2_data, cot_length_map)
        
        # Calculate overall proportions
        total_neg1 = sum(r['total_negative'] for r in results1)
        total_rew1 = sum(r['total_rewards'] for r in results1)
        overall_prop1 = (total_neg1 / total_rew1 * 100) if total_rew1 > 0 else 0
        
        total_neg2 = sum(r['total_negative'] for r in results2)
        total_rew2 = sum(r['total_rewards'] for r in results2)
        overall_prop2 = (total_neg2 / total_rew2 * 100) if total_rew2 > 0 else 0
        
        model1_results.append(results1)
        model2_results.append(results2)
        model1_overall_props.append(overall_prop1)
        model2_overall_props.append(overall_prop2)
    
    # Average overall proportions
    avg_prop1 = np.mean(model1_overall_props)
    avg_prop2 = np.mean(model2_overall_props)
    
    print(f"\nOverall Results:")
    print(f"{args.model1_name}: {avg_prop1:.2f}% ± {np.std(model1_overall_props):.2f}% negative rewards")
    print(f"{args.model2_name}: {avg_prop2:.2f}% ± {np.std(model2_overall_props):.2f}% negative rewards")
    
    # Create bins based on the first run
    bin_edges = create_length_bins(model1_results[0], args.num_bins)
    
    # Compute average performance by length bins
    print("Computing negative proportions by CoT length...")
    
    # Average performance across runs for each bin
    model1_bin_props = [[] for _ in range(len(bin_edges) - 1)]
    model2_bin_props = [[] for _ in range(len(bin_edges) - 1)]
    
    for run_idx in range(args.num_runs):
        m1_perf = compute_negative_proportion_by_length(model1_results[run_idx], bin_edges)
        m2_perf = compute_negative_proportion_by_length(model2_results[run_idx], bin_edges)
        
        for bin_idx in range(len(bin_edges) - 1):
            model1_bin_props[bin_idx].append(m1_perf[bin_idx]['overall_negative_proportion'])
            model2_bin_props[bin_idx].append(m2_perf[bin_idx]['overall_negative_proportion'])
    
    # Create averaged performance data
    model1_avg_performance = []
    model2_avg_performance = []
    
    for bin_idx in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        bin_center = (bin_min + bin_max) / 2
        
        # Use first run for count (should be consistent)
        count = compute_negative_proportion_by_length(model1_results[0], bin_edges)[bin_idx]['count']
        
        model1_avg_performance.append({
            'bin_range': (bin_min, bin_max),
            'overall_negative_proportion': np.mean(model1_bin_props[bin_idx]),
            'count': count,
            'bin_center': bin_center
        })
        
        model2_avg_performance.append({
            'bin_range': (bin_min, bin_max),
            'overall_negative_proportion': np.mean(model2_bin_props[bin_idx]),
            'count': count,
            'bin_center': bin_center
        })
    
    # Create comparison plot
    print("Creating comparison plot...")
    plot_negative_proportion_comparison(model1_avg_performance, model2_avg_performance, 
                                      args.model1_name, args.model2_name, 
                                      avg_prop1, avg_prop2, args.num_runs, args.output_plot)
    
    # Print detailed bin results
    print("\nDetailed Results by CoT Length Bins (Equal Sample Size):")
    print(f"{'Length Range':<15} {'Count':<8} {args.model1_name + ' Neg%':<12} {args.model2_name + ' Neg%':<12} {'Difference':<12}")
    print("-" * 65)
    
    for i in range(len(model1_avg_performance)):
        bin_range = f"{int(model1_avg_performance[i]['bin_range'][0])}-{int(model1_avg_performance[i]['bin_range'][1])}"
        count = model1_avg_performance[i]['count']
        prop1 = model1_avg_performance[i]['overall_negative_proportion']
        prop2 = model2_avg_performance[i]['overall_negative_proportion']
        diff = prop1 - prop2
        
        print(f"{bin_range:<15} {count:<8} {prop1:<12.1f} {prop2:<12.1f} {diff:<12.1f}")

if __name__ == "__main__":
    main()