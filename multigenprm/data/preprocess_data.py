import os
import re
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoTokenizer
from multigenprm.utils import parse_orm_label, parse_prm_label, trim_after_first_verdict, truncate_after_last_boxed_step

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

def get_orm_label(data):
    if (data["parsed_answer"] is None) or (data["answer"] is None):
        return -1 if -1 in data["labels"] else 1
    else:
        return 1 if data["parsed_answer"] == data["answer"] else -1

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
    
    # Make all labels before first error position become 1
    for i in range(0, first_error_pos):
        normalized[i] = 1
    
    # Make all labels after first error position become -1
    for i in range(first_error_pos, len(normalized)):
        normalized[i] = -1
    
    return normalized

def get_prm_label(data):
    """Get PRM labels with normalization."""
    # Use normalized original labels
    full_labels = normalize_process_labels(data["labels"])
    
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
        if args.task_type == "orm": 
            self.processor = {
                "preprocess": trim_after_first_verdict,
                "parse label": parse_orm_label,
                "get label": get_orm_label,
                "parse condition": lambda parsed_label, K: not np.isnan(parsed_label),
                "correctness condition": lambda parsed_label, label: parsed_label == label,
                "get yes_or_no": lambda parsed_label: None,
                "format": lambda critique, yes_or_no: f"<think>\nLet's verify step by step:{critique}",
                "process cot": lambda cot, label: cot,
                "check positive": lambda label: label == 1,
                "check negative": lambda label: label == -1
            }
        else:
            self.processor = {
                "preprocess": truncate_after_last_boxed_step,
                "parse label": parse_prm_label,
                "get label": get_prm_label,
                "parse condition": lambda parsed_label, K: is_valid_label_format(parsed_label, K),
                "correctness condition": lambda parsed_label, label: np.array_equal(parsed_label, label),
                "get yes_or_no": lambda parsed_label: "No" if -1 in parsed_label else "Yes",
                "format": lambda critique, yes_or_no: f"<think>\nLet's verify step by step:{critique}\nIs the solution correct? {yes_or_no}",
                "process cot": lambda cot, label: cot[:label.index(-1)+1] if -1 in label else cot,
                "check positive": lambda label: all([l==1 for l in label]),
                "check negative": lambda label: -1 in label
            }
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    def _process_single_example(self, data, seen_critiques):
        # process critique
        critique = self.processor["preprocess"](data["critique"])
             
        # Check for duplicates
        if critique in seen_critiques:
            return None, "duplicate"
        
        # Check for </think> token
        if not ("</think>" in critique):
            return None, "think"
        
        # Check for Chinese characters
        if bool(re.search(r'[\u4e00-\u9fff]', critique)):
            return None, "chinese"
        
        # Parse and get labels
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
        
        # Check token length
        if self.args.max_tokens > 0:
            tokenized = self.tokenizer(critique, add_special_tokens=False)
            critique_length = len(tokenized["input_ids"])
            if critique_length > self.args.max_tokens:
                return None, "length"
        
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
            # "critique_length": critique_length,
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
                limited_examples.append(sampled)
        
        # Flatten the list if needed
        flat_limited_examples = []
        for item in limited_examples:
            if isinstance(item, list):
                flat_limited_examples.extend(item)
            else:
                flat_limited_examples.append(item)
        
        print(f"After limiting: {len(flat_limited_examples)} examples (reduced from {len(examples)})")
        
        return flat_limited_examples
    
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
        input_path = os.path.join(self.args.input_dir, f"{self.args.category}_dataset.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
        if self.args.debug:
            dataset = dataset[:1000]
            
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
        pos_count = sum(1 for ex in examples if self.processor["check positive"](ex["label"]))
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
        
        output_path = os.path.join(
            self.args.output_dir, 
            f"preprocessed_{self.args.category}_dataset.json"
        )
        
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=4)
        
        print(f"Results are saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other', 'prm800k', 'all'],
                       help="Category of problems to process")
    parser.add_argument("--task_type", type=str, default="orm", choices=['orm', 'prm'])
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducible balancing")
    parser.add_argument("--max_tokens", type=int, default=0, 
                       help="Number of max tokens")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Total number of samples in final balanced dataset (will be split equally between positive and negative)")
    parser.add_argument("--num_unique_ids", type=int, default=None,
                       help="Maximum number of critiques per unique (q_id, cot_id) pair. If set to 1, randomly selects one critique per pair")
    parser.add_argument("--no_balance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print("Using same output directory as input directory")
    
    categories = (['law', 'psychology', 'chemistry', 'biology', 'physics', 
                   'history', 'economics', 'math', 'business', 'philosophy', 
                   'health', 'engineering', 'computer_science', 'other'] 
                  if args.category == 'all' else [args.category])
    
    for category in categories:
        args.category = category
        status_parts = [f"seed: {args.seed}"]
        if args.num_samples is not None:
            status_parts.append(f"num_samples: {args.num_samples}")
        if args.num_unique_ids is not None:
            status_parts.append(f"num_unique_ids: {args.num_unique_ids}")
        
        status = " with " + ", ".join(status_parts)
        print(f"Processing category: {category}{status}")
        
        preprocessor = DatasetPreprocessor(args)
        preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()