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
    for i in range(len(normalized)):
        if i < first_error_pos:
            normalized[i] = 1
        else:
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
        self.get_prm_label_with_noise = lambda data: get_prm_label(
            data, self.args.noise_data_ratio, self.args.noise_process_ratio, self.args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    def _process_single_example(self, data):
        label = self.get_prm_label_with_noise(data)
        formatted_critique = ""
        
        yes_or_no = "Yes"
        for step_idx, l in enumerate(label):
            if l == 1:
                formatted_critique += f"\nStep {step_idx+1}: The step is " + "\\boxed{correct} "
            elif l == -1:
                formatted_critique += f"\nStep {step_idx+1}: The step is " + "\\boxed{incorrect} "
                yes_or_no = "No"
                break
            else:
                raise NotImplementedError
        formatted_critique += f"\nIs the solution correct? {yes_or_no}"
                
        return {
            "q_id": data["q_id"],    
            "question": data["question"],
            "cot_id": data["cot_id"],         
            "cot": data["cot"][:len(label)],
            "critique": formatted_critique,
            "label": label,
        }
    
    def preprocess_dataset(self):
        """Main preprocessing function"""
        # Load dataset
        input_path = os.path.join(self.args.input_dir, "prm800k_dataset.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
        if self.args.debug:
            dataset = dataset[:1000]
        
        examples = []
        # Process each example
        for data in tqdm(dataset):
            if 'labels' not in data or \
                data['labels'] is None or \
                    len(data['labels']) == 0:
                continue
            
            processed = self._process_single_example(data)
            
            examples.append(processed)
        
        self._save_results(examples)        
        return examples
    
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
    if args.noise_data_ratio > 0.0:
        status_parts.append(f"noise_data_ratio: {args.noise_data_ratio}")
        status_parts.append(f"noise_process_ratio: {args.noise_process_ratio}")
    
    status = " with " + ", ".join(status_parts)
    print(f"Processing PRM800k dataset{status}")
    
    preprocessor = DatasetPreprocessor(args)
    preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()