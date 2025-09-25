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

import numpy as np

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

def get_first_error_step_index(labels):
    """Helper function to find first error position"""
    for i, label in enumerate(labels):
        if label == -1:
            return i
    return len(labels)

def normalize_process_labels(labels):
    """
    Normalize labels to valid process format: [1,1,1,...,-1,-1,-1,...]
    Once we find the first -1, all subsequent labels become -1.
    """
    if not labels:
        return []
    
    normalized = labels.copy()
    first_error_pos = get_first_error_step_index(labels)
    
    # Make all labels after first error position become -1
    for i in range(first_error_pos, len(normalized)):
        normalized[i] = -1
    
    return normalized

def get_deterministic_rng_for_pair(q_id, cot_id, seed):
    """
    Create a deterministic random number generator for a specific (q_id, cot_id) pair.
    This ensures the same pair always gets the same noise pattern.
    """
    # Create a deterministic seed based on q_id, cot_id, and global seed
    combined_string = f"{q_id}_{cot_id}_{seed}"
    hash_object = hashlib.md5(combined_string.encode())
    deterministic_seed = int(hash_object.hexdigest()[:8], 16)  # Use first 8 hex chars as seed
    return random.Random(deterministic_seed)

def should_apply_noise_to_pair(q_id, cot_id, noise_data_ratio, seed):
    """
    Deterministically decide whether to apply noise to a specific (q_id, cot_id) pair.
    This ensures all data points with the same (q_id, cot_id) get the same treatment.
    """
    if noise_data_ratio == 0.0:
        return False
    
    pair_rng = get_deterministic_rng_for_pair(q_id, cot_id, seed)
    return pair_rng.random() < noise_data_ratio

def apply_process_noise_deterministic(original_labels, noise_process_ratio, q_id, cot_id, seed):
    """
    Apply process-aware noise to labels with deterministic behavior for same (q_id, cot_id) pairs.
    
    Args:
        original_labels: Original labels
        noise_process_ratio: Proportion of process steps to flip (0.0 to 1.0)
        q_id: Question ID
        cot_id: Chain of thought ID
        seed: Global seed for deterministic behavior
    
    Returns:
        Noisy labels maintaining process constraint
    """
    K = len(original_labels)
    normalized_original = normalize_process_labels(original_labels)
    
    if noise_process_ratio == 0.0:
        return normalized_original
    
    # Use deterministic RNG for this specific (q_id, cot_id) pair
    pair_rng = get_deterministic_rng_for_pair(q_id, cot_id, seed)
    
    # Calculate target number of flips
    target_flips = int(K * noise_process_ratio)
    if target_flips == 0:
        return normalized_original
    
    # Find the first error position in normalized labels
    original_first_error = get_first_error_step_index(normalized_original)
    
    # Generate all possible valid label configurations
    possible_configs = []
    
    for first_error_pos in range(K + 1):
        config = [1] * first_error_pos + [-1] * (K - first_error_pos)
        
        # Calculate how many labels differ from normalized original
        flips = sum(1 for orig, new in zip(normalized_original, config) if orig != new)
        
        possible_configs.append({
            'config': config,
            'first_error_pos': first_error_pos, 
            'flips': flips
        })
    
    # Remove the original configuration to ensure we actually apply noise
    possible_configs = [c for c in possible_configs if c['first_error_pos'] != original_first_error]
    
    # Filter configs that match our target flip count exactly
    target_configs = [c for c in possible_configs if c['flips'] == target_flips]
    
    # If no exact match, choose the closest match
    if not target_configs:
        closest_flips = min(possible_configs, key=lambda c: abs(c['flips'] - target_flips))['flips']
        target_configs = [c for c in possible_configs if c['flips'] == closest_flips]
    
    # Randomly select from valid configurations using deterministic RNG
    selected_config = pair_rng.choice(target_configs)
    
    return selected_config['config']

def get_prm_label(data, noise_data_ratio=0.0, noise_process_ratio=0.0, seed=42):
    """Get PRM labels with optional noise injection."""
    # Use normalized original labels
    full_labels = normalize_process_labels(data["labels"])
    
    # Apply noise if requested
    if noise_data_ratio > 0.0:
        q_id = data.get("q_id", "unknown")
        cot_id = data.get("cot_id", "unknown")
        
        if should_apply_noise_to_pair(q_id, cot_id, noise_data_ratio, seed):
            full_labels = apply_process_noise_deterministic(full_labels, noise_process_ratio, q_id, cot_id, seed)
    
    # Cut off at the first -1 (original behavior)
    label = []
    for l in full_labels:
        label.append(l)
        if l == -1:
            break    
    
    return label

class DatasetPreprocessor:
    def __init__(self, args):
        self.args = args
        
        # Create a closure that captures the noise parameters
        def get_prm_label_with_noise(data):
            return get_prm_label(data, self.args.noise_data_ratio, self.args.noise_process_ratio, self.args.seed)
        
        self.processor = {
            "preprocess": truncate_after_last_boxed_step,
            "parse label": parse_prm_label,
            "get label": get_prm_label_with_noise,
            "parse condition": lambda parsed_label, K: is_valid_label_format(parsed_label, K),
            "correctness condition": lambda parsed_label, label: np.array_equal(parsed_label, label),
            "get yes_or_no": lambda parsed_label: "No" if -1 in parsed_label else "Yes",
            "format": lambda critique, yes_or_no: f"<think>\nLet's verify step by step:{critique}\nIs the solution correct? {yes_or_no}",
            "process cot": lambda cot, label: cot[:label.index(-1)+1] if -1 in label else cot,
            "check positive": lambda label: all([l==1 for l in label]),
            "check negative": lambda label: -1 in label
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        
        # Track noise application for statistics
        self.noise_applied_pairs = set()
        self.total_pairs = set()
    
    def _process_single_example(self, data, seen_critiques):
        # Track unique pairs for noise statistics
        if self.args.noise_data_ratio > 0.0:
            q_id = data.get("q_id", "unknown")
            cot_id = data.get("cot_id", "unknown")
            self.total_pairs.add((q_id, cot_id))
            
            if should_apply_noise_to_pair(q_id, cot_id, self.args.noise_data_ratio, self.args.seed):
                self.noise_applied_pairs.add((q_id, cot_id))
        
        # process critique
        critique = self.processor["preprocess"](data["critique"])
             
        # Check for duplicates
        if critique in seen_critiques:
            return None, "duplicate"
        
        # Check for </think> token
        if not re.search(r"</think>\n\n(.*)", critique, re.DOTALL):
            return None, "think"
        
        # Check for Chinese characters
        if bool(re.search(r'[\u4e00-\u9fff]', critique)):
            return None, "chinese"
        
        # Parse and get labels (now with potential noise injection)
        parsed_label = self.processor["parse label"](critique)
        label = self.processor["get label"](data)
        
        # 0 cot
        if len(data["cot"]) == 0:
            return None, "0 cot"
        
        # not parsable
        if not self.processor["parse condition"](parsed_label, len(data["cot"])):
            return None, "not parsable"
        
        # correctness        
        if not self.processor["correctness condition"](parsed_label, label):
            return None, "incorrect label"
        
        # Add to seen critiques
        seen_critiques.add(critique)
        
        # postprocess
        formatted_critique = self.processor["format"](critique, self.processor["get yes_or_no"](parsed_label))
                
        return {
            "q_id": data["q_id"],            
            "question": data["question"],
            "cot_id": data["cot_id"],             
            "cot": self.processor["process cot"](data["cot"], label),
            "critique": formatted_critique,
            "label": parsed_label,
        }, None
    
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
                # If group has fewer examples than limit, keep all
                limited_examples.extend(group_examples)
            else:
                # Randomly sample from the group
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
            # Use num_samples to control the total dataset size
            samples_per_class = self.args.num_samples // 2
            max_possible_per_class = min(len(positive_examples), len(negative_examples))
            
            if samples_per_class > max_possible_per_class:
                print(f"Warning: Requested {samples_per_class} samples per class, but only {max_possible_per_class} available per class")
                samples_per_class = max_possible_per_class
            
            print(f"Using num_samples={self.args.num_samples}: {samples_per_class} samples per class")
        else:
            # Original behavior: use minimum of both classes
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
        """Main preprocessing function"""
        # Load dataset
        input_path = os.path.join(self.args.input_dir, "prm800k_dataset.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
        if self.args.debug:
            dataset = dataset[:1000]
            
        if self.args.noise_data_ratio <= 0 or self.args.noise_process_ratio <= 0:
            self._save_results(dataset)
            return None
            
        # Track statistics
        skip_counts = {
            "duplicate": 0, "think": 0, "chinese": 0, "0 cot": 0,
            "not parsable": 0, "incorrect label": 0, "length": 0
        }
        
        valid_examples = []
        seen_critiques = set()
        
        # Process each example
        for data in tqdm(dataset):
            if 'labels' not in data or \
                data['labels'] is None or \
                    len(data['labels']) == 0:
                continue
            
            processed, skip_reason = self._process_single_example(data, seen_critiques)

            if processed is None:
                skip_counts[skip_reason] += 1
                continue
            
            valid_examples.append(processed)
        
        # Print noise application statistics
        if self.args.noise_data_ratio > 0.0:
            print(f"\n=== Noise Application Statistics ===")
            print(f"Total unique (q_id, cot_id) pairs: {len(self.total_pairs)}")
            print(f"Pairs with noise applied: {len(self.noise_applied_pairs)}")
            if len(self.total_pairs) > 0:
                print(f"Actual noise application rate: {len(self.noise_applied_pairs)/len(self.total_pairs):.3f}")
            print(f"Target noise_data_ratio: {self.args.noise_data_ratio:.3f}")
            print(f"Noise_process_ratio: {self.args.noise_process_ratio:.3f}")
        
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
        self._save_results(final_examples)
        
        # Print first example for verification
        if final_examples:
            print("\n=== Sample Example ===")
            print(final_examples[0]["critique"])
        
        return final_examples
    
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
    parser = argparse.ArgumentParser(description="Preprocess PRM800k dataset")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible balancing")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Total number of samples in final balanced dataset (will be split equally between positive and negative)")
    parser.add_argument("--num_unique_ids", type=int, default=None,
                       help="Maximum number of critiques per unique (q_id, cot_id) pair. If set to 1, randomly selects one critique per pair")
    parser.add_argument("--no_balance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    # Noise injection arguments
    parser.add_argument("--noise_data_ratio", type=float, default=0.0,
                       help="Proportion of unique (q_id, cot_id) pairs that will be noisified (0.0 to 1.0)")
    parser.add_argument("--noise_process_ratio", type=float, default=0.0,
                       help="Proportion of process steps to flip for each noisified pair (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print("Using same output directory as input directory")
    
    # Validate noise arguments
    if args.noise_data_ratio < 0.0 or args.noise_data_ratio > 1.0:
        raise ValueError("noise_data_ratio must be between 0.0 and 1.0")
    if args.noise_process_ratio < 0.0 or args.noise_process_ratio > 1.0:
        raise ValueError("noise_process_ratio must be between 0.0 and 1.0")
    
    status_parts = [f"seed: {args.seed}"]
    if args.num_samples is not None:
        status_parts.append(f"num_samples: {args.num_samples}")
    if args.num_unique_ids is not None:
        status_parts.append(f"num_unique_ids: {args.num_unique_ids}")
    if args.noise_data_ratio > 0.0:
        status_parts.append(f"noise_data_ratio: {args.noise_data_ratio}")
        status_parts.append(f"noise_process_ratio: {args.noise_process_ratio}")
    
    status = " with " + ", ".join(status_parts)
    print(f"Processing PRM800k dataset{status}")
    
    preprocessor = DatasetPreprocessor(args)
    preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()