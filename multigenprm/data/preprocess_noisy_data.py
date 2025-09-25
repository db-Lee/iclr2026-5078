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
    if any(np.isnan(x) for x in label):
        return False
    if all(x == 1 for x in label):
        return len(label) == K
    if label and label[-1] == -1:
        if label.count(-1) != 1:
            return False
        if not all(x == 1 for x in label[:-1]):
            return False
        return len(label) <= K
    return False

def normalize_process_labels(labels):
    if not labels:
        return []
    normalized = labels.copy()
    first_error_pos = next((i for i, label in enumerate(labels) if label == -1), len(labels))
    for i in range(first_error_pos):
        normalized[i] = 1
    for i in range(first_error_pos, len(normalized)):
        normalized[i] = -1
    return normalized

def get_prm_label(data):
    full_labels = normalize_process_labels(data["labels"])
    label = []
    for l in full_labels:
        label.append(l)
        if l == -1:
            break    
    return label

def should_apply_noise_to_pair(q_id, cot_id, noise_data_ratio, seed):
    if noise_data_ratio == 0.0:
        return False
    combined_string = f"{q_id}_{cot_id}_{seed}"
    hash_object = hashlib.md5(combined_string.encode())
    deterministic_seed = int(hash_object.hexdigest()[:8], 16)
    rng = random.Random(deterministic_seed)
    return rng.random() < noise_data_ratio

def has_all_ones(labels):
    return all(x == 1 for x in labels)

def has_negative_one(labels):
    return -1 in labels

class DatasetPreprocessor:
    def __init__(self, args):
        self.args = args
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
        self.valid_critiques_cache = {}
    
    def _preprocess_all_critiques(self, dataset):
        seen_critiques = set()
        for data in tqdm(dataset, desc="Caching critiques"):
            if 'labels' not in data or data['labels'] is None or len(data['labels']) == 0:
                continue
            
            critique = self.processor["preprocess"](data["critique"])
            if critique in seen_critiques or not ("</think>" in critique):
                continue
            if bool(re.search(r'[\u4e00-\u9fff]', critique)) or len(data["cot"]) == 0:
                continue
            
            parsed_label = self.processor["parse label"](critique)
            if not self.processor["parse condition"](parsed_label, len(data["cot"])):
                continue
            
            if hasattr(self.args, 'max_tokens') and self.args.max_tokens > 0:
                tokenized = self.tokenizer(critique, add_special_tokens=False)
                if len(tokenized["input_ids"]) > self.args.max_tokens:
                    continue
            
            seen_critiques.add(critique)
            pair_key = (data["q_id"], data["cot_id"])
            if pair_key not in self.valid_critiques_cache:
                self.valid_critiques_cache[pair_key] = []
            self.valid_critiques_cache[pair_key].append({
                'parsed_label': parsed_label,
                'original_data': data,
                'critique': critique,
                'has_all_ones': has_all_ones(parsed_label),
                'has_negative_one': has_negative_one(parsed_label)
            })
    
    def _create_final_example(self, valid_data, is_noisy=False):
        yes_or_no = self.processor["get yes_or_no"](valid_data["parsed_label"])
        formatted_critique = self.processor["format"](valid_data["critique"], yes_or_no)
        return {
            "q_id": valid_data["original_data"]["q_id"],            
            "question": valid_data["original_data"]["question"],
            "cot_id": valid_data["original_data"]["cot_id"],             
            "cot": self.processor["process cot"](valid_data["original_data"]["cot"], valid_data["parsed_label"]),
            "critique": formatted_critique,
            "label": valid_data["parsed_label"],
            "is_noisy": is_noisy
        }
    
    def _balance_examples(self, examples):
        random.seed(self.args.seed)
        
        noisy_positive = [ex for ex in examples if self.processor["check positive"](ex["label"]) and ex.get("is_noisy", False)]
        noisy_negative = [ex for ex in examples if self.processor["check negative"](ex["label"]) and ex.get("is_noisy", False)]
        clean_positive = [ex for ex in examples if self.processor["check positive"](ex["label"]) and not ex.get("is_noisy", False)]
        clean_negative = [ex for ex in examples if self.processor["check negative"](ex["label"]) and not ex.get("is_noisy", False)]
        
        if hasattr(self.args, 'num_samples') and self.args.num_samples is not None:
            target_per_class = self.args.num_samples // 2
        else:
            total_positive = len(noisy_positive) + len(clean_positive)
            total_negative = len(noisy_negative) + len(clean_negative)
            target_per_class = min(total_positive, total_negative)
        
        random.shuffle(noisy_positive)
        random.shuffle(noisy_negative)
        random.shuffle(clean_positive)
        random.shuffle(clean_negative)
        
        def select_with_noise_priority(noisy_list, clean_list, target_count):
            selected = []
            noisy_count = min(len(noisy_list), target_count)
            selected.extend(noisy_list[:noisy_count])
            remaining_slots = target_count - noisy_count
            if remaining_slots > 0:
                clean_count = min(len(clean_list), remaining_slots)
                selected.extend(clean_list[:clean_count])
            return selected, noisy_count, len(selected) - noisy_count
        
        selected_positive, pos_noisy_count, pos_clean_count = select_with_noise_priority(
            noisy_positive, clean_positive, target_per_class)
        selected_negative, neg_noisy_count, neg_clean_count = select_with_noise_priority(
            noisy_negative, clean_negative, target_per_class)
        
        balanced_examples = selected_positive + selected_negative
        random.shuffle(balanced_examples)
        
        total_noisy = pos_noisy_count + neg_noisy_count
        total_clean = pos_clean_count + neg_clean_count
        
        print(f"# critiques after balancing: {len(balanced_examples)}")
        print(f"  Positive: {len(selected_positive)} | Negative: {len(selected_negative)}")
        print(f"  Noisy: {total_noisy} ({total_noisy/len(balanced_examples):.1%}) | Clean: {total_clean} ({total_clean/len(balanced_examples):.1%})")
        
        return balanced_examples
    
    def preprocess_dataset(self):
        input_path = os.path.join(self.args.input_dir, "prm800k_dataset.json")
        with open(input_path, "r") as f:
            dataset = json.load(f)
        if self.args.debug:
            dataset = dataset[:1000]
        
        print(f"# critiques: {len(dataset)}")
        
        # Step 1: Cache all valid critiques
        self._preprocess_all_critiques(dataset)
        
        # Step 2: Process unique pairs
        noisy_examples, clean_examples, total_pairs = [], [], set()
        
        for data in tqdm(dataset, desc="Processing pairs"):
            if 'labels' not in data or data['labels'] is None or len(data['labels']) == 0:
                continue
            
            q_id, cot_id = data["q_id"], data["cot_id"]
            pair_key = (q_id, cot_id)
            
            if pair_key in total_pairs:
                continue
            total_pairs.add(pair_key)
            
            if pair_key not in self.valid_critiques_cache:
                continue
            
            original_label = normalize_process_labels(data["labels"])
            original_cut_label = []
            for l in original_label:
                original_cut_label.append(l)
                if l == -1:
                    break
            
            should_flip = should_apply_noise_to_pair(q_id, cot_id, self.args.noise_data_ratio, self.args.seed)
            
            if should_flip:
                parsed_label = None
                for valid_data in self.valid_critiques_cache[pair_key]:
                    include = False
                    if has_all_ones(original_cut_label) and valid_data["has_negative_one"]:
                        if parsed_label is None:                            
                            parsed_label = valid_data['parsed_label']
                            include = True
                        else:
                            include = np.array_equal(valid_data['parsed_label'], parsed_label)
                            
                    elif has_negative_one(original_cut_label) and valid_data["has_all_ones"]:
                        include = True
                    
                    if include:
                        example = self._create_final_example(valid_data, is_noisy=True)
                        noisy_examples.append(example)
            else:
                for valid_data in self.valid_critiques_cache[pair_key]:
                    if np.array_equal(valid_data['parsed_label'], original_cut_label):
                        example = self._create_final_example(valid_data, is_noisy=False)
                        clean_examples.append(example)
        
        all_examples = clean_examples + noisy_examples
        total_noisy = len(noisy_examples)
        total_clean = len(clean_examples)
        
        print(f"# critiques after preprocessing: {len(all_examples)}")
        print(f"  Positive: {sum(1 for ex in all_examples if self.processor['check positive'](ex['label']))}")
        print(f"  Negative: {sum(1 for ex in all_examples if self.processor['check negative'](ex['label']))}")
        print(f"  Noisy: {total_noisy} ({total_noisy/len(all_examples):.1%}) | Clean: {total_clean} ({total_clean/len(all_examples):.1%})")
        
        # Balance examples
        if not self.args.no_balance:
            final_examples = self._balance_examples(all_examples)
        else:
            final_examples = all_examples
        
        # Save results
        clean_final_examples = []
        for example in final_examples:
            clean_example = example.copy()
            clean_example.pop("is_noisy", None)
            clean_final_examples.append(clean_example)
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        output_path = os.path.join(self.args.output_dir, "preprocessed_prm800k_dataset.json")
        with open(output_path, "w") as f:
            json.dump(clean_final_examples, f, indent=4)
        
        return clean_final_examples

def main():
    parser = argparse.ArgumentParser(description="Preprocess PRM800k dataset with noise injection")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_balance", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--num_unique_ids", type=int, default=None)
    parser.add_argument("--noise_data_ratio", type=float, default=0.0,
                       help="Proportion of pairs that will have labels flipped (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    if args.noise_data_ratio < 0.0 or args.noise_data_ratio > 1.0:
        raise ValueError("noise_data_ratio must be between 0.0 and 1.0")
    
    print(f"Processing PRM800k with noise_data_ratio={args.noise_data_ratio}")
    
    preprocessor = DatasetPreprocessor(args)
    preprocessor.preprocess_dataset()

if __name__ == "__main__":
    main()