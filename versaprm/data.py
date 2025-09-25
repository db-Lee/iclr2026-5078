import os
import json
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy

def merge_dicts(dict_list):
    merged = deepcopy(dict_list[0])
    for d in dict_list[1:]:
        for k, v in d.items():
            merged[k].extend(v)
    return merged

def tokenize_step(step, label, tokenizer, mask_id=-100, label_last_n=None):
    tokenized = tokenizer(step, add_special_tokens=False)
    step_len = len(tokenized.input_ids)
    
    if label_last_n is None or label_last_n >= step_len:
        labels = [label] * step_len
    else:
        labels = [mask_id] * (step_len - label_last_n) + [label] * label_last_n
    
    tokenized['labels'] = labels
    return tokenized

def get_first_error_step_index(labels):
    for i, label in enumerate(labels):
        if label == -1:
            return i
    return len(labels)

def normalize_process_labels(labels):
    """
    Normalize labels to valid process format: [1,1,1,...,-1,-1,-1,...]
    Once we find the first -1, all subsequent labels become -1.
    """
    normalized = labels.copy()
    first_error_pos = get_first_error_step_index(labels)
    
    # Make all labels after first error position become -1
    for i in range(first_error_pos, len(normalized)):
        normalized[i] = -1
    
    return normalized

def apply_process_noise(original_labels, noise_process_ratio, rng):
    """
    Apply process-aware noise to labels.
    
    Args:
        original_labels: Original labels
        noise_process_ratio: Proportion of process steps to flip (0.0 to 1.0)
        rng: Random number generator
    
    Returns:
        Noisy labels maintaining process constraint
    """
    K = len(original_labels)
    normalized_original = normalize_process_labels(original_labels)
    
    if noise_process_ratio == 0.0:
        return normalized_original
    
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
    
    # Randomly select from valid configurations
    selected_config = rng.choice(target_configs)
    
    return selected_config['config']

def load_dataset(data_path, category):
    category_list = ['law', 'psychology', 'chemistry', 'biology', 
                     'physics', 'history', 'economics', 'math', 'business', 
                     'philosophy', 'health', 'engineering', 'computer_science', 'other']
    if category == "prm800k":
        category_list = ["prm800k"]
    elif category == "all":
        pass
    elif category in category_list:
        category_list = [category]
    else:
        raise NotImplementedError
    
    dataset = []
    for cat in category_list:
        with open(os.path.join(data_path, f"{cat}_dataset.json"), "r") as f:
            dataset.extend(json.load(f))
    return dataset

def tokenize_one_data(data, tokenizer, mask_id=-100, label_last_n=None, max_length=None, orm=False, 
                     labels=None):
    """Tokenize one data point with given labels."""
    # Question
    question_tok = tokenizer(f"{data['question']} \n\n")
    question_tok['labels'] = [mask_id] * len(question_tok.input_ids)
    
    if labels is None:
        return []
    
    steps_tok = []
    
    # Process steps
    for i, step in enumerate(data['cot']):
        if orm:
            # ORM: only predict on last step
            if (data["parsed_answer"] is None) or (data["answer"] is None):
                label = 0 if -1 in labels else 1
            else:
                label = 1 if data["parsed_answer"] == data["answer"] else 0
            step_label_last_n = label_last_n if i == len(data['cot']) - 1 else 0
        else:
            # PRM: predict on all steps
            label = 0 if labels[i] == -1 else 1
            step_label_last_n = label_last_n
            
        step_tok = tokenize_step(f'{step} \n\n\n\n', label, tokenizer, mask_id, step_label_last_n)
        steps_tok.append(step_tok)
        
        # PRM: stop at first incorrect step
        if not orm and label == 0:
            break
    
    tokenized = merge_dicts([question_tok] + steps_tok)
    return tokenized if max_length is None or len(tokenized.input_ids) <= max_length else None

def balance_dataset(total_tokenized, total_K, rng, orm=True):
    assert len(total_tokenized) == len(total_K)
    positive_examples = []
    negative_examples = []
    
    for data, K in zip(total_tokenized, total_K):
        labels = data["labels"]
        
        if orm:
            if labels[-1] == 1:
                positive_examples.append(data)
            else:
                negative_examples.append(data)
        else:
            positive_count = sum(1 for label in labels if label == 1)
            if positive_count == K:
                positive_examples.append(data)
            else:
                negative_examples.append(data)
    
    # Balance by taking minimum count of both
    min_count = min(len(positive_examples), len(negative_examples))
    
    # Randomly sample equal numbers from both categories
    balanced_positive = rng.sample(positive_examples, min_count)
    balanced_negative = rng.sample(negative_examples, min_count)
    
    # Combine and shuffle
    balanced_data = balanced_positive + balanced_negative
    rng.shuffle(balanced_data)
    
    return balanced_data

def tokenize_dataset(data_path, category, tokenizer, mask_id=-100, label_last_n=None, 
                    max_length=None, task_type="prm", noise_data_ratio=0.0, noise_process_ratio=0.0, 
                    seed=42, no_balance=False):
    """
    Main tokenization function with two-parameter noise control and balancing.
    
    Args:
        task_type: Either "prm" or "orm"
        noise_data_ratio: Proportion of data samples that will be noisified (0.0 to 1.0)
        noise_process_ratio: Proportion of process steps to flip for each noisified sample (0.0 to 1.0)
        no_balance: If True, skip balancing positive and negative examples
    """
    dataset = load_dataset(data_path, category)
    rng = random.Random(seed)
    
    # Convert task_type to orm boolean for compatibility
    orm = (task_type == "orm")
    
    total_tokenized, total_K = [], []
    for data in tqdm(dataset):
        if 'labels' not in data or \
            data['labels'] is None or \
                len(data['labels']) == 0:
            continue
        
        labels = data['labels']
        
        # Apply noise based on two parameters (only for PRM)
        if task_type == "prm":
            if rng.random() < noise_data_ratio:
                labels = apply_process_noise(labels, noise_process_ratio, rng)
        
        tokenized = tokenize_one_data(data, tokenizer, mask_id, label_last_n, 
                                         max_length, orm, labels)
        if tokenized is not None:
            total_tokenized.append(tokenized)
            total_K.append(len(labels))
    
    # Balance dataset if requested (default behavior)
    if not no_balance:
        total_tokenized = balance_dataset(total_tokenized, total_K, rng, orm)
    
    return total_tokenized

class TokenizedPRMDataset(Dataset):
    def __init__(self, data_path, category, tokenizer, label_mask_token_id=-100, 
                 label_last_n=None, max_length=None, task_type="prm", 
                 noise_data_ratio=0.0, noise_process_ratio=0.0, seed=None, no_balance=False):
        """
        Args:
            task_type: Either "prm" or "orm"
            noise_data_ratio: Proportion of data samples that will be noisified (0.0 to 1.0)
            noise_process_ratio: Proportion of process steps to flip for each noisified sample (0.0 to 1.0)
            no_balance: If True, skip balancing positive and negative examples (default: False)
        """
        super().__init__()
        assert task_type in ["prm", "orm"]
        seed = 42 if seed is None else seed
        self.tokenized_data = tokenize_dataset(data_path, category, tokenizer, 
                                             label_mask_token_id, label_last_n, max_length, 
                                             task_type, noise_data_ratio, noise_process_ratio, 
                                             seed, no_balance)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, i):
        return self.tokenized_data[i]

# Example usage:
# Balanced dataset (default):
# dataset = TokenizedPRMDataset(
#     data_path="path/to/data", 
#     category="math",
#     tokenizer=tokenizer,
#     task_type="prm",           # or "orm"
#     noise_data_ratio=0.3,      # 30% of data will be noisified
#     noise_process_ratio=0.5,   # For noisified data, flip 50% of process steps
#     seed=42
# )

# Unbalanced dataset:
# dataset = TokenizedPRMDataset(
#     data_path="path/to/data", 
#     category="math",
#     tokenizer=tokenizer,
#     task_type="prm",
#     no_balance=True            # Skip balancing
# )