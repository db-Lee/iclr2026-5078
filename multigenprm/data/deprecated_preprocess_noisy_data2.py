import os
import re
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import hashlib

from transformers import AutoTokenizer
from multigenprm.utils import parse_prm_label, truncate_after_last_boxed_step

def is_valid_label_format(label, K):
    # Check 1: No np.nan
    if any(np.isnan(x) for x in label):
        return False

    # Case 1: all 1s
    if all(x == 1 for x in label):
        return len(label) == K

    # Case 2: ends with exactly one -1, and all before it are 1
    if label and label[-1] == -1:
        if label.count(-1) != 1:
            return False
        if not all(x == 1 for x in label[:-1]):
            return False
        return len(label) <= K

    return False

def normalize_process_labels(labels):
    """
    Normalize labels to valid process format: [1,1,1,...,-1,-1,-1,...]
    Once we find the first -1, all subsequent labels become -1.
    """
    if not labels:
        return []
    
    normalized = labels.copy()
    first_error_pos = None
    
    for i, label in enumerate(labels):
        if label == -1:
            first_error_pos = i
            break
    
    if first_error_pos is not None:
        for i in range(first_error_pos, len(normalized)):
            normalized[i] = -1
    
    return normalized

def should_apply_noise_to_pair(q_id, cot_id, noise_data_ratio, seed):
    """
    Deterministically decide whether to apply noise to a specific (q_id, cot_id) pair.
    """
    if noise_data_ratio == 0.0:
        return False
    
    # Create deterministic seed based on q_id, cot_id, and global seed
    combined_string = f"{q_id}_{cot_id}_{seed}"
    hash_object = hashlib.md5(combined_string.encode())
    deterministic_seed = int(hash_object.hexdigest()[:8], 16)
    
    rng = random.Random(deterministic_seed)
    return rng.random() < noise_data_ratio

def has_all_ones(labels):
    """Check if all labels are 1s"""
    return all(x == 1 for x in labels)

def has_negative_one(labels):
    """Check if labels contain -1"""
    return -1 in labels

class DatasetPreprocessor:
    def __init__(self, args):
        self.args = args
        
        self.processor = {
            "preprocess": truncate_after_last_boxed_step,
            "parse label": parse_prm_label,
            "parse condition": lambda parsed_label, K: is_valid_label_format(parsed_label, K),
            "get yes_or_no": lambda parsed_label: "No" if -1 in parsed_label else "Yes",
            "format": lambda critique, yes_or_no: f"<think>\nLet's verify step by step:{critique}\nIs the solution correct? {yes_or_no}",
            "process cot": lambda cot, label: cot[:label.index(-1)+1] if -1 in label else cot,
            "check positive": lambda label: all([l==1 for l in label]),
            "check negative": lambda label: -1 in label
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Caches for faster processing
        self.valid_critiques_cache = {}  # critique_text -> {parsed_label, q_id, cot_id, original_data}
        self.critiques_by_label_type = {
            "all_ones": [],      # critiques that have all 1s
            "has_negative": []   # critiques that have -1 somewhere
        }
        
        # Track noise application for statistics
        self.noise_applied_pairs = set()
        self.total_pairs = set()
        self.original_vs_final_labels = []  # Track (original_label_type, final_label_type, was_flipped)
    
    def _preprocess_all_critiques(self, dataset):
        """
        First pass: preprocess all critiques and cache their parsed labels.
        Only keep critiques that are valid (ignoring label matching condition).
        """
        print("Preprocessing and caching all critiques...")
        
        seen_critiques = set()
        
        for data in tqdm(dataset, desc="Processing critiques"):
            if 'labels' not in data or data['labels'] is None or len(data['labels']) == 0:
                continue
            
            # Preprocess critique
            critique = self.processor["preprocess"](data["critique"])
            
            # Skip if already seen
            if critique in seen_critiques:
                continue
            
            # Basic validation checks
            if not re.search(r"</think>\n\n(.*)", critique, re.DOTALL):
                continue
            
            if bool(re.search(r'[\u4e00-\u9fff]', critique)):
                continue
            
            if len(data["cot"]) == 0:
                continue
            
            # Parse labels from critique
            parsed_label = self.processor["parse label"](critique)
            
            # Check if parsable
            if not self.processor["parse condition"](parsed_label, len(data["cot"])):
                continue
            
            # Cache this valid critique
            seen_critiques.add(critique)
            self.valid_critiques_cache[critique] = {
                'parsed_label': parsed_label,
                'q_id': data["q_id"],
                'cot_id': data["cot_id"],
                'original_data': data
            }
            
            # Categorize by label type for fast lookup
            if has_all_ones(parsed_label):
                self.critiques_by_label_type["all_ones"].append(critique)
            elif has_negative_one(parsed_label):
                self.critiques_by_label_type["has_negative"].append(critique)
        
        print(f"Cached {len(self.valid_critiques_cache)} valid critiques")
        print(f"- All ones: {len(self.critiques_by_label_type['all_ones'])}")
        print(f"- Has negative: {len(self.critiques_by_label_type['has_negative'])}")
    
    def _find_matching_critique(self, target_has_all_ones):
        """
        Fast lookup for critique with target label type.
        """
        if target_has_all_ones:
            return self.critiques_by_label_type["all_ones"]
        else:
            return self.critiques_by_label_type["has_negative"]
    
    def _create_final_example(self, original_data, critique_text, parsed_label):
        """Create final example from original data and (possibly flipped) critique"""
        
        # Format critique (compute on the fly - simple and reliable)
        yes_or_no = self.processor["get yes_or_no"](parsed_label)
        formatted_critique = self.processor["format"](critique_text, yes_or_no)
        
        # Get original labels for COT processing
        original_labels = normalize_process_labels(original_data["labels"])
        original_cut_labels = []
        for l in original_labels:
            original_cut_labels.append(l)
            if l == -1:
                break
        
        return {
            "q_id": original_data["q_id"],            
            "question": original_data["question"],
            "cot_id": original_data["cot_id"],             
            "cot": self.processor["process cot"](original_data["cot"], original_cut_labels),
            "critique": formatted_critique,
            "label": parsed_label,
        }
    
    def _limit_examples_per_unique_id(self, examples):
        """Limit the number of examples per unique (q_id, cot_id) pair"""
        if self.args.num_unique_ids is None:
            return examples
        
        print(f"\nLimiting to {self.args.num_unique_ids} examples per unique (q_id, cot_id) pair")
        
        # Group examples by (q_id, cot_id)
        grouped_examples = defaultdict(list)
        for example in examples:
            key = (example["q_id"], example["cot_id"])
            grouped_examples[key].append(example)
        
        print(f"Found {len(grouped_examples)} unique (q_id, cot_id) pairs")
        
        # Set random seed for reproducible sampling
        random.seed(self.args.seed)
        
        # Sample from each group
        limited_examples = []
        for key, group_examples in grouped_examples.items():
            if len(group_examples) <= self.args.num_unique_ids:
                limited_examples.extend(group_examples)
            else:
                random.shuffle(group_examples)
                sampled = group_examples[:self.args.num_unique_ids]
                limited_examples.extend(sampled)
        
        print(f"After limiting: {len(limited_examples)} examples (reduced from {len(examples)})")
        return limited_examples
    
    def _balance_examples(self, examples):
        # Set random seed for reproducibility
        random.seed(self.args.seed)
        
        # Separate positive and negative examples
        positive_examples = []
        negative_examples = []
        
        for example in examples:
            if self.processor["check positive"](example["label"]):
                positive_examples.append(example)
            elif self.processor["check negative"](example["label"]):
                negative_examples.append(example)
            else:
                raise NotImplementedError            
        
        print(f"\n=== Original Distribution Before Balancing ===")
        print(f"Total positive examples: {len(positive_examples)}")
        print(f"Total negative examples: {len(negative_examples)}")
        
        # Determine the size for balanced dataset
        if self.args.num_samples is not None:
            samples_per_class = self.args.num_samples // 2
            max_possible_per_class = min(len(positive_examples), len(negative_examples))
            
            if samples_per_class > max_possible_per_class:
                print(f"Warning: Requested {samples_per_class} samples per class, but only {max_possible_per_class} available per class")
                samples_per_class = max_possible_per_class
            
            print(f"Using num_samples={self.args.num_samples}: {samples_per_class} samples per class")
        else:
            samples_per_class = min(len(positive_examples), len(negative_examples))
            print(f"Balancing to {samples_per_class} examples per class (original behavior)")
        
        # Randomly sample from each class
        random.shuffle(positive_examples)
        random.shuffle(negative_examples)
        
        balanced_positive = positive_examples[:samples_per_class]
        balanced_negative = negative_examples[:samples_per_class]
        
        # Combine and shuffle the final dataset
        balanced_examples = balanced_positive + balanced_negative
        random.shuffle(balanced_examples)
        
        print(f"Final balanced dataset: {len(balanced_examples)} examples")
        print(f"Positive: {len(balanced_positive)}, Negative: {len(balanced_negative)}")
        
        return balanced_examples
    
    def preprocess_dataset(self):
        """Main preprocessing function with cached mandatory flipping"""
        # Load dataset
        input_path = os.path.join(self.args.input_dir, "prm800k_dataset.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
        if self.args.debug:
            dataset = dataset[:1000]
        
        # Step 1: Preprocess and cache all valid critiques
        self._preprocess_all_critiques(dataset)
        
        # Step 2: Process each data point, using cached critiques
        print("\nProcessing final examples...")
        
        skip_counts = {
            "no_original_critique": 0,
            "no_flip_candidate": 0,
            "no_valid_critique": 0
        }
        
        valid_examples = []
        
        for data in tqdm(dataset, desc="Creating final examples"):
            if 'labels' not in data or data['labels'] is None or len(data['labels']) == 0:
                continue
            
            q_id = data.get("q_id", "unknown")
            cot_id = data.get("cot_id", "unknown")
            pair_key = (q_id, cot_id)
            self.total_pairs.add(pair_key)
            
            # Get original labels
            original_labels = normalize_process_labels(data["labels"])
            original_cut_labels = []
            for l in original_labels:
                original_cut_labels.append(l)
                if l == -1:
                    break
            
            # Check if we should apply noise to this pair
            should_flip = should_apply_noise_to_pair(q_id, cot_id, self.args.noise_data_ratio, self.args.seed)
            
            if should_flip:
                self.noise_applied_pairs.add(pair_key)
                
                # Find flip candidate from cache
                original_has_all_ones = has_all_ones(original_cut_labels)
                flip_candidates = self._find_matching_critique(
                    target_has_all_ones=not original_has_all_ones
                )
                
                if not flip_candidates:
                    skip_counts["no_flip_candidate"] += 1
                    continue
                
                # Randomly select a flip candidate
                random.seed(self.args.seed + hash(pair_key))
                selected_critique = random.choice(flip_candidates)
                selected_parsed_label = self.valid_critiques_cache[selected_critique]['parsed_label']
                
                # Create example with flipped critique
                example = self._create_final_example(data, selected_critique, selected_parsed_label)
                valid_examples.append(example)
                
                # Track noise effect
                original_type = "all_ones" if original_has_all_ones else "has_negative"
                final_type = "all_ones" if has_all_ones(selected_parsed_label) else "has_negative"
                self.original_vs_final_labels.append((original_type, final_type, True))
                
            else:
                # Use original critique if it's in our valid cache
                original_critique = self.processor["preprocess"](data["critique"])
                
                if original_critique not in self.valid_critiques_cache:
                    skip_counts["no_valid_critique"] += 1
                    continue
                
                original_parsed_label = self.valid_critiques_cache[original_critique]['parsed_label']
                
                # Verify correctness (original behavior)
                if not np.array_equal(original_parsed_label, original_cut_labels):
                    skip_counts["no_valid_critique"] += 1
                    continue
                
                # Create example with original critique
                example = self._create_final_example(data, original_critique, original_parsed_label)
                valid_examples.append(example)
                
                # Track no noise effect
                original_type = "all_ones" if has_all_ones(original_cut_labels) else "has_negative"
                final_type = "all_ones" if has_all_ones(original_parsed_label) else "has_negative"
                self.original_vs_final_labels.append((original_type, final_type, False))
        
        # Print noise application statistics
        if self.args.noise_data_ratio > 0.0:
            print(f"\n=== Noise Application Statistics ===")
            print(f"Total unique (q_id, cot_id) pairs: {len(self.total_pairs)}")
            print(f"Pairs with noise applied: {len(self.noise_applied_pairs)}")
            if len(self.total_pairs) > 0:
                print(f"Actual noise application rate: {len(self.noise_applied_pairs)/len(self.total_pairs):.3f}")
            print(f"Target noise_data_ratio: {self.args.noise_data_ratio:.3f}")
        
        print(f"Valid examples before limiting per unique ID: {len(valid_examples)}")
        
        # Limit examples per unique (q_id, cot_id) pair
        limited_examples = self._limit_examples_per_unique_id(valid_examples)
        
        print(f"Valid examples after limiting per unique ID: {len(limited_examples)}")
        
        # Balance the examples
        if self.args.no_balance:
            final_examples = limited_examples
        else:
            final_examples = self._balance_examples(limited_examples)
        
        self._print_statistics(len(dataset), len(final_examples), skip_counts, final_examples)
        self._print_noise_analysis(final_examples)
        self._save_results(final_examples)
        
        # Print first example for verification
        if final_examples:
            print("\n=== Sample Example ===")
            print(final_examples[0]["critique"])
        
        return final_examples
    
    def _print_noise_analysis(self, final_examples):
        """Print detailed analysis of noise effects after balancing"""
        if self.args.noise_data_ratio <= 0.0:
            return
            
        print(f"\n" + "="*60)
        print(f"DETAILED NOISE ANALYSIS AFTER BALANCING")
        print(f"="*60)
        
        # Count examples by original vs final label types
        noise_stats = {
            ("all_ones", "all_ones", True): 0,    # Original all_ones -> Final all_ones (flipped)
            ("all_ones", "has_negative", True): 0, # Original all_ones -> Final has_negative (flipped)
            ("has_negative", "all_ones", True): 0, # Original has_negative -> Final all_ones (flipped)  
            ("has_negative", "has_negative", True): 0, # Original has_negative -> Final has_negative (flipped)
            ("all_ones", "all_ones", False): 0,   # Original all_ones -> Final all_ones (not flipped)
            ("all_ones", "has_negative", False): 0, # Original all_ones -> Final has_negative (not flipped)
            ("has_negative", "all_ones", False): 0, # Original has_negative -> Final all_ones (not flipped)
            ("has_negative", "has_negative", False): 0, # Original has_negative -> Final has_negative (not flipped)
        }
        
        # Map final examples back to their noise tracking data
        example_to_noise_data = {}
        for i, (orig_type, final_type, was_flipped) in enumerate(self.original_vs_final_labels):
            if i < len(final_examples):
                example_to_noise_data[i] = (orig_type, final_type, was_flipped)
        
        # Count noise effects in final balanced dataset
        for i, example in enumerate(final_examples):
            if i in example_to_noise_data:
                orig_type, final_type, was_flipped = example_to_noise_data[i]
                noise_stats[(orig_type, final_type, was_flipped)] += 1
        
        total_final = len(final_examples)
        total_flipped = sum(count for (orig, final, flipped), count in noise_stats.items() if flipped)
        total_not_flipped = sum(count for (orig, final, flipped), count in noise_stats.items() if not flipped)
        
        print(f"Total examples in final dataset: {total_final}")
        print(f"Examples with noise applied: {total_flipped} ({total_flipped/total_final:.1%})")
        print(f"Examples without noise: {total_not_flipped} ({total_not_flipped/total_final:.1%})")
        
        print(f"\n--- FLIPPED EXAMPLES (Noise Applied) ---")
        flipped_stats = {k: v for k, v in noise_stats.items() if k[2] == True and v > 0}
        if flipped_stats:
            for (orig_type, final_type, _), count in flipped_stats.items():
                pct = count / total_final * 100
                flip_direction = f"{orig_type} → {final_type}"
                print(f"  {flip_direction:25} {count:4d} examples ({pct:5.1f}%)")
        else:
            print("  No flipped examples in final dataset")
        
        print(f"\n--- NON-FLIPPED EXAMPLES (Original Labels) ---")
        not_flipped_stats = {k: v for k, v in noise_stats.items() if k[2] == False and v > 0}
        if not_flipped_stats:
            for (orig_type, final_type, _), count in not_flipped_stats.items():
                pct = count / total_final * 100
                direction = f"{orig_type} → {final_type}"
                print(f"  {direction:25} {count:4d} examples ({pct:5.1f}%)")
        else:
            print("  No non-flipped examples in final dataset")
        
        # Label correctness analysis
        print(f"\n--- FINAL LABEL DISTRIBUTION ---")
        final_positive = sum(1 for ex in final_examples if has_all_ones(ex["label"]))
        final_negative = total_final - final_positive
        print(f"  Positive (all 1s):     {final_positive:4d} examples ({final_positive/total_final:.1%})")
        print(f"  Negative (has -1):     {final_negative:4d} examples ({final_negative/total_final:.1%})")
        
        # Expected vs actual flip success
        if total_flipped > 0:
            successful_flips = (noise_stats[("all_ones", "has_negative", True)] + 
                              noise_stats[("has_negative", "all_ones", True)])
            failed_flips = total_flipped - successful_flips
            
            print(f"\n--- FLIP SUCCESS RATE ---")
            print(f"  Successful flips:      {successful_flips:4d} examples ({successful_flips/total_flipped:.1%} of flipped)")
            print(f"  Failed flips:          {failed_flips:4d} examples ({failed_flips/total_flipped:.1%} of flipped)")
            
            if failed_flips > 0:
                print(f"  └─ Note: Failed flips happen when no opposite-label critique was available")
        
        print(f"="*60)
    
    def _print_statistics(self, total, valid_count, skip_counts, examples):
        """Print preprocessing statistics"""
        print(f"Final examples after preprocessing and balancing: {valid_count} / {total}")
        for skip_type, count in skip_counts.items():
            print(f"# {skip_type} skipped: {count}")
        
        # Label distribution
        pos_count = sum(1 for ex in examples if "Is the solution correct? Yes" in ex["critique"])
        neg_count = len(examples) - pos_count
        
        if examples:
            print(f"\n=== Label Distribution ===")
            print(f"Positive examples (correct): {pos_count} ({pos_count/len(examples):.2%})")
            print(f"Negative examples (incorrect): {neg_count} ({neg_count/len(examples):.2%})")
            print(f"Total examples: {len(examples)}")
            
            # Print (q_id, cot_id) diversity statistics
            unique_pairs = set((ex["q_id"], ex["cot_id"]) for ex in examples)
            print(f"\n=== Diversity Statistics ===")
            print(f"Unique (q_id, cot_id) pairs: {len(unique_pairs)}")
            print(f"Examples per unique pair (avg): {len(examples) / len(unique_pairs):.2f}")
            if self.args.num_unique_ids is not None:
                print(f"Maximum examples per unique pair: {self.args.num_unique_ids}")
        else:
            print("\n=== Label Distribution ===\nNo examples found")
    
    def _save_results(self, examples):
        """Save preprocessed examples to file"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        filename = "preprocessed_prm800k_dataset.json"
        output_path = os.path.join(self.args.output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=4)
        
        print(f"Results are saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess PRM800k dataset with cached mandatory label flipping")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible processing")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Total number of samples in final balanced dataset")
    parser.add_argument("--num_unique_ids", type=int, default=None,
                       help="Maximum number of critiques per unique (q_id, cot_id) pair")
    parser.add_argument("--no_balance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    # Simplified noise injection argument
    parser.add_argument("--noise_data_ratio", type=float, default=0.0,
                       help="Proportion of unique (q_id, cot_id) pairs that will have their labels mandatorily flipped (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print("Using same output directory as input directory")
    
    # Validate noise arguments
    if args.noise_data_ratio < 0.0 or args.noise_data_ratio > 1.0:
        raise ValueError("noise_data_ratio must be between 0.0 and 1.0")
    
    status_parts = [f"seed: {args.seed}"]
    if args.num_samples is not None:
        status_parts.append(f"num_samples: {args.num_samples}")
    if args.num_unique_ids is not None:
        status_parts.append(f"num_unique_ids: {args.num_unique_ids}")
    if args.noise_data_ratio > 0.0:
        status_parts.append(f"noise_data_ratio (mandatory flip): {args.noise_data_ratio}")
    
    status = " with " + ", ".join(status_parts)
    print(f"Processing PRM800k dataset{status}")
    
    preprocessor = DatasetPreprocessor(args)
    preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()