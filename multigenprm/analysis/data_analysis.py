import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Visualize histogram of token lengths for critiques")
    
    parser.add_argument("--data_dir", type=str, default="./local_datasets/train")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Path to save figure and statistics")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)      
    category_list = ['law', 'psychology', 'chemistry', 'biology', 'physics', 'history', 'economics', 'math', 'business', 'philosophy', 'health', 'engineering', 'computer_science', 'other']
    
    # Store results for plotting
    categories = []
    question_proportions = []
    cot_proportions = []
    distinct_cots_counts = []
    critiques_counts = []
    critique_lengths = []
    
    for category in tqdm(category_list, desc="Processing categories"):
        # Load preprocessed dataset
        dataset_path = os.path.join(args.input_dir, f"preprocessed_{category}_dataset.json")
        print(f"Loading dataset from: {dataset_path}")
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"Warning: File not found: {dataset_path}, skipping...")
            continue
            
        with open(dataset_path, 'r') as f:
            preprocessed_dataset = json.load(f)     
        
        with open(os.path.join(args.data_dir, f"{category}_dataset.json"), "r") as f:
            original_dataset = json.load(f)
        
        # Get original statistics
        original_distinct_questions = set([d["q_id"] for d in original_dataset])
        original_distinct_cots = set([(d["q_id"], d["cot_id"]) for d in original_dataset])
            
        # Get preprocessed statistics and critique lengths
        preprocessed_distinct_questions = set()
        preprocessed_distinct_cots = set()
        category_critique_lengths = []
        
        for d in preprocessed_dataset:
            preprocessed_distinct_questions.add(d["q_id"])
            preprocessed_distinct_cots.add((d["q_id"], d["cot_id"]))
            
            # Calculate critique length in tokens
            if "critique_length" not in d or args.tokenize:
                critique_tokens = tokenizer.encode(d["critique"])
                critique_length = len(critique_tokens)
                # critique_length = len(d["critique"])
                category_critique_lengths.append(critique_length)
            else:
                category_critique_lengths.append(d["critique_length"])
                
        total_critiques = len(preprocessed_dataset)
        
        # Calculate proportions
        question_proportion = len(preprocessed_distinct_questions) / len(original_distinct_questions)
        cot_proportion = len(preprocessed_distinct_cots) / len(original_distinct_cots)
        
        # Store results
        categories.append(category)
        question_proportions.append(question_proportion)
        cot_proportions.append(cot_proportion)
        distinct_cots_counts.append(len(preprocessed_distinct_cots))
        critiques_counts.append(total_critiques)
        critique_lengths.extend(category_critique_lengths)  # Add all critique lengths to global list
        
        print(f"{category}:")
        print(f"  Questions: {len(preprocessed_distinct_questions)}/{len(original_distinct_questions)} (proportion: {question_proportion:.3f})")
        print(f"  CoTs: {len(preprocessed_distinct_cots)}/{len(original_distinct_cots)} (proportion: {cot_proportion:.3f})")
        print(f"  Distinct CoTs: {len(preprocessed_distinct_cots)}, Critiques: {total_critiques}")
        if category_critique_lengths:
            print(f"  Critique lengths - Mean: {np.mean(category_critique_lengths):.1f}, Median: {np.median(category_critique_lengths):.1f}, Max: {max(category_critique_lengths)}")
        
        # Clean up memory
        del original_dataset
        del preprocessed_dataset
        gc.collect()
    
    # Create four side-by-side plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Proportion of preprocessed questions over original questions
    bars1 = ax1.bar(categories, question_proportions, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Proportion of Preprocessed Questions', fontsize=12)
    ax1.set_title('Preprocessed Questions / Original Questions', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, prop in zip(bars1, question_proportions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prop:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Proportion of preprocessed CoTs over original CoTs
    bars2 = ax2.bar(categories, cot_proportions, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Proportion of Preprocessed CoTs', fontsize=12)
    ax2.set_title('Preprocessed CoTs / Original CoTs', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, prop in zip(bars2, cot_proportions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prop:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Absolute numbers (distinct CoTs vs critiques)
    x = np.arange(len(categories))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, distinct_cots_counts, width, label='Distinct CoTs', 
                     color='lightgreen', alpha=0.7, edgecolor='black')
    bars3b = ax3.bar(x + width/2, critiques_counts, width, label='Critiques', 
                     color='gold', alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Absolute Numbers: Distinct CoTs vs Critiques', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45)
    ax3.legend()
    
    # Add value labels on bars for plot 3
    max_count = max(max(distinct_cots_counts), max(critiques_counts))
    for bar, val in zip(bars3a, distinct_cots_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max_count*0.01,
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars3b, critiques_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max_count*0.01,
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Histogram of critique lengths
    if critique_lengths:
        ax4.hist(critique_lengths, bins=50, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Critique Length (tokens)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Critique Lengths', fontsize=14)
        
        # Add statistics text
        mean_length = np.mean(critique_lengths)
        median_length = np.median(critique_lengths)
        max_length = max(critique_lengths)
        min_length = min(critique_lengths)
        
        stats_text = f'Mean: {mean_length:.1f}\nMedian: {median_length:.1f}\nMin: {min_length}\nMax: {max_length}\nTotal: {len(critique_lengths)}'
        ax4.text(0.7, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No critique data available', transform=ax4.transAxes, 
                fontsize=14, ha='center', va='center')
        ax4.set_title('Distribution of Critique Lengths', fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plots
    output_path = os.path.join(args.output_dir, "cot_analysis_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    # Save statistics to JSON
    stats_data = {
        "categories": categories,
        "question_proportions": question_proportions,
        "cot_proportions": cot_proportions,
        "distinct_cots_counts": distinct_cots_counts,
        "critiques_counts": critiques_counts,
        "critique_lengths_stats": {
            "mean": float(np.mean(critique_lengths)) if critique_lengths else 0,
            "median": float(np.median(critique_lengths)) if critique_lengths else 0,
            "min": int(min(critique_lengths)) if critique_lengths else 0,
            "max": int(max(critique_lengths)) if critique_lengths else 0,
            "total_critiques": len(critique_lengths)
        },
        "summary": {
            "total_categories_processed": len(categories),
            "average_question_proportion": np.mean(question_proportions),
            "average_cot_proportion": np.mean(cot_proportions),
            "total_distinct_cots": sum(distinct_cots_counts),
            "total_critiques": sum(critiques_counts)
        }
    }
    
    stats_path = os.path.join(args.output_dir, "cot_analysis_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=4)
    print(f"Statistics saved to: {stats_path}")
    
    # Show the plot if not in headless mode
    try:
        plt.show()
    except:
        print("Display not available, plot saved to file only.")

if __name__ == "__main__":
    main()