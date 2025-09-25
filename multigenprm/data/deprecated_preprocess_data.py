import os
import re
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

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
    if "parsed_answer" in data and "answer" in data:
        return 1 if data["parsed_answer"] == data["answer"] else -1
    else:
        return -1 if -1 in data["labels"] else 1

def get_prm_label(data):
    label = []
    for l in data["labels"]:
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
        if not re.search(r"</think>\n\n(.*)", critique, re.DOTALL):
            return None, "think"
        
        # Check for Chinese characters
        if bool(re.search(r'[\u4e00-\u9fff]', critique)):
            return None, "chinese"
        
        # Parse and fix labels
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
        tokenized = self.tokenizer(critique, add_special_tokens=False)
        critique_length = len(tokenized["input_ids"])
        if critique_length > 4096:
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
            "critique_length": critique_length,
            "label": parsed_label,
        }, None
    
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
        
        # Determine the size for balanced dataset (min of both classes)
        min_class_size = min(len(positive_examples), len(negative_examples))
        
        print(f"Balancing to {min_class_size} examples per class")
        
        # Randomly sample from each class
        random.shuffle(positive_examples)
        random.shuffle(negative_examples)
        
        balanced_positive = positive_examples[:min_class_size]
        balanced_negative = negative_examples[:min_class_size]
        
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
            
        # Track statistics
        skip_counts = {
            "duplicate": 0, "think": 0, "chinese": 0, "0 cot": 0,
            "not parsable": 0, "incorrect label": 0, "length": 0
        }
        
        valid_examples = []
        seen_critiques = set()
        
        # Process each example
        for data in tqdm(dataset):
            processed, skip_reason = self._process_single_example(data, seen_critiques)

            if processed is None:
                skip_counts[skip_reason] += 1
                continue
            
            valid_examples.append(processed)
        
        print(f"Valid examples before balancing: {len(valid_examples)}")
        
        # Balance the examples
        balanced_examples = self._balance_examples(valid_examples)
        
        self._print_statistics(len(dataset), len(balanced_examples), skip_counts, balanced_examples)
        self._save_results(balanced_examples)
        
        # Print first example for verification
        if balanced_examples:
            print("\n=== Sample Example ===")
            print(balanced_examples[0]["critique"])
        
        return balanced_examples
    
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
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
        print("Using same output directory as input directory")
    
    categories = (['law', 'psychology', 'chemistry', 'biology', 'physics', 
                   'history', 'economics', 'math', 'business', 'philosophy', 
                   'health', 'engineering', 'computer_science', 'other', 'prm800k'] 
                  if args.category == 'all' else [args.category])
    
    for category in categories:
        args.category = category
        print(f"Processing category: {category} with seed: {args.seed}")
        preprocessor = DatasetPreprocessor(args)
        preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()
    