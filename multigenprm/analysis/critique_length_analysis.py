import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import gc

def main():
    parser = argparse.ArgumentParser(description="Visualize histogram of token lengths for critiques")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to checkpoint directory containing tokenizer and eval dataset")
    parser.add_argument("--input_dir", type=str, default=None, 
                       help="Path to load critiques")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Path to save figure and statistics")
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other'],
                       help="Category of problems to analyze")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Batch size for tokenization to manage memory usage")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    
    # Load dataset
    dataset_path = os.path.join(args.input_dir, f"eval_{args.category}_dataset.json")
    print(f"Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Collect all critiques
    all_critiques = []
    for data_item in dataset:
        critiques = data_item.get("critiques", [])
        for critique_list in critiques:
            all_critiques.extend(critique_list)
    
    # Filter out empty critiques
    all_critiques = [text for text in all_critiques if text]
    
    print(f"Total critiques found: {len(all_critiques)}")
    
    if len(all_critiques) == 0:
        print("No critiques found in the dataset!")
        return
    
    # Process in batches to manage memory
    print(f"Tokenizing critiques in batches of {args.batch_size}...")
    token_lengths = []
    
    for i in tqdm(range(0, len(all_critiques), args.batch_size)):
        batch_end = min(i + args.batch_size, len(all_critiques))
        batch_critiques = all_critiques[i:batch_end]
        
        # Tokenize batch
        encoded = tokenizer(batch_critiques, add_special_tokens=True, return_length=True)
        batch_lengths = encoded['length']
        
        # Only keep the lengths, discard the encoded tokens
        token_lengths.extend(batch_lengths)
        
        # Explicitly delete the encoded data and force garbage collection
        del encoded
        del batch_critiques
        gc.collect()
    
    # Clear the original critiques list to free memory
    del all_critiques
    gc.collect()
    
    print(f"Total critiques processed: {len(token_lengths)}")
    
    # Calculate statistics
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    std_length = np.std(token_lengths)
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)
    
    print(f"\nToken Length Statistics:")
    print(f"Mean: {mean_length:.2f}")
    print(f"Median: {median_length:.2f}")
    print(f"Std Dev: {std_length:.2f}")
    print(f"Min: {min_length}")
    print(f"Max: {max_length}")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    n_bins = min(50, max(10, len(set(token_lengths))))  # Adaptive bin count
    plt.hist(token_lengths, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for statistics
    plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    plt.axvline(median_length, color='green', linestyle='--', linewidth=2, label=f'Median: {median_length:.1f}')
    
    # Formatting
    plt.xlabel('Token Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Token Lengths in Critiques\nCategory: {args.category.title()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Count: {len(token_lengths)}\nMean: {mean_length:.1f}\nMedian: {median_length:.1f}\nStd: {std_length:.1f}\nRange: {min_length}-{max_length}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot to checkpoint_path
    output_path = os.path.join(args.checkpoint_path, f"token_length_histogram_{args.category}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")
    
    # Show the plot if not in headless mode
    try:
        plt.show()
    except:
        print("Display not available, plot saved to file only.")
    
    # Save detailed statistics to file by default
    stats_path = os.path.join(args.checkpoint_path, f"token_length_stats_{args.category}.json")
    stats_data = {
        "category": args.category,
        "total_critiques": len(token_lengths),
        "mean_length": float(mean_length),
        "median_length": float(median_length),
        "std_length": float(std_length),
        "min_length": int(min_length),
        "max_length": int(max_length),
        "token_lengths": token_lengths  # Full list for further analysis
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=4)
    print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()