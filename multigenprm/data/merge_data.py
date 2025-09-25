import argparse
import json
import os
from datasets import load_dataset
from collections import defaultdict

category_list = ['law', 'psychology', 'chemistry', 'biology', 'physics', 'history', 'economics', 'math', 'business', 'philosophy', 'health', 'engineering', 'computer_science', 'other']

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--input_dirs", type=str, required=True, nargs='+',
                       help="Input directories (space separated)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="law", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', 'all'
    }, help="Category of problems to process")
    args = parser.parse_args()
    
    # Input directories are already a list with nargs='+'
    input_dirs = args.input_dirs
    
    # Filter categories based on the argument
    if args.category == "all":
        categories_to_process = category_list
    else:
        categories_to_process = [args.category]
        
    for category in categories_to_process:
        print(f"Processing category: {category}")
        
        # Start with the first dataset as base - keep ALL entries including multiple critiques
        first_dir = input_dirs[0]
        first_dataset_path = os.path.join(first_dir, f"preprocessed_{category}_dataset.json")
        
        if not os.path.exists(first_dataset_path):
            print(f"Warning: {first_dataset_path} does not exist, starting with empty dataset")
            merged_dataset = []
            seen_qid_cot_pairs = set()
        else:
            print(f"Loading base dataset: {first_dataset_path}")
            with open(first_dataset_path, "r") as f:
                merged_dataset = json.load(f)
            
            # Track all (q_id, cot_id) pairs we've seen
            seen_qid_cot_pairs = set()
            for d in merged_dataset:
                seen_qid_cot_pairs.add((d["q_id"], d["cot_id"]))
        
        # Process remaining input directories
        for input_dir in input_dirs[1:]:
            dataset_path = os.path.join(input_dir, f"preprocessed_{category}_dataset.json")
            
            if not os.path.exists(dataset_path):
                print(f"Warning: {dataset_path} does not exist, skipping")
                continue
                
            print(f"Processing: {dataset_path}")
            
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
            
            # First pass: identify all new (q_id, cot_id) pairs in this dataset
            new_qid_cot_pairs = set()
            for d in dataset:
                pair = (d["q_id"], d["cot_id"])
                if pair not in seen_qid_cot_pairs:
                    new_qid_cot_pairs.add(pair)
            
            # Second pass: add ALL entries (critiques) for the new (q_id, cot_id) pairs
            for d in dataset:
                pair = (d["q_id"], d["cot_id"])
                if pair in new_qid_cot_pairs:
                    merged_dataset.append(d)
            
            # Update seen pairs with the new ones
            seen_qid_cot_pairs.update(new_qid_cot_pairs)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save the merged dataset
        output_path = os.path.join(args.output_dir, f"preprocessed_{category}_dataset.json")
        with open(output_path, "w") as f:
            json.dump(merged_dataset, f, indent=2)
        
        print(f"Merged dataset saved to: {output_path}")
        print(f"Total distinct entries for {category}: {len(merged_dataset)}")

if __name__ == "__main__":
    main()