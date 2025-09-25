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

def construct_labels_from_error_index(K, first_error_index):
    if first_error_index >= K:
        return [1] * K
    else:
        return [1] * first_error_index + [-1] * (K - first_error_index)

def corrupt_process_labels(original_labels, rng):
    # """
    # Noise logic:
    # 1) if -1 in labels -> make all labels 1 (remove errors)
    # 2) else -> make somewhere -1 (introduce error)
    # """
    # if -1 in original_labels:
    #     # Case 1: Has errors -> remove all errors (all 1s)
    #     return [1] * len(original_labels)
    # else:
    #     # Case 2: No errors -> introduce error at random position
    #     K = len(original_labels)
    #     error_position = rng.randint(0, K-1)  # Random position from 0 to K-1
    #     return [1] * error_position + [-1] * (K - error_position)
    
    K = len(original_labels)
    possible_steps = [ _ for _ in range(0, K+1) ]
    first_error_step = get_first_error_step_index(original_labels)
    possible_steps.pop(first_error_step)
    if len(possible_steps) == 0:
        return original_labels
    noisy_first_error_step = rng.choice(possible_steps)
    return construct_labels_from_error_index(K, noisy_first_error_step)    

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

def get_tokenized_length(data, tokenizer, labels):
    """Calculate the tokenized length for given data and labels without actually tokenizing."""
    # Question length
    question_tok = tokenizer(f"{data['question']} \n\n", add_special_tokens=False)
    total_length = len(question_tok.input_ids)
    
    # Process steps
    for i, step in enumerate(data['cot']):
        # PRM: stop at first incorrect step
        label = 1 if labels[i] == 1 else 0
        step_tok = tokenizer(f'{step} \n\n\n\n', add_special_tokens=False)
        total_length += len(step_tok.input_ids)
        
        if label == 0:  # Stop at first incorrect step
            break
    
    return total_length

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
            if data["parsed_answer"] is not None and data["answer"] is not None:
                label = 1 if data["parsed_answer"] == data["answer"] else 0
            else:
                label = 0 if -1 in labels else 1
            step_label_last_n = label_last_n if i == len(data['cot']) - 1 else 0
        else:
            # PRM: predict on all steps
            label = 1 if labels[i] == 1 else 0
            step_label_last_n = label_last_n
            
        step_tok = tokenize_step(f'{step} \n\n\n\n', label, tokenizer, mask_id, step_label_last_n)
        steps_tok.append(step_tok)
        
        # PRM: stop at first incorrect step
        if not orm and label == 0:
            break
    
    tokenized = merge_dicts([question_tok] + steps_tok)
    return [tokenized] if max_length is None or len(tokenized.input_ids) <= max_length else []

def tokenize_dataset(data_path, category, tokenizer, mask_id=-100, label_last_n=None, 
                    max_length=None, orm=False, noise_ratio=0.0, seed=None):
    """Main tokenization function with proper noise ratio preservation."""
    dataset = load_dataset(data_path, category)
    rng = random.Random(seed) if seed is not None else None
    
    if noise_ratio == 0.0 or orm:
        # No noise case or ORM (where length doesn't depend on labels) - original behavior
        tokenized = []
        for data in tqdm(dataset):
            if 'labels' not in data or data['labels'] is None:
                continue
            
            labels = data['labels']
            if not orm and noise_ratio > 0 and rng is not None and rng.random() < noise_ratio:
                labels = corrupt_process_labels(labels, rng)
            
            tokenized.extend(tokenize_one_data(data, tokenizer, mask_id, label_last_n, 
                                             max_length, orm, labels))
        return tokenized
    
    # PRM with noise case - need to preserve noise ratio after length-based filtering
    valid_data = []
    
    # Step 1: Collect all data that could be valid (check both original and noisy versions)
    for data in tqdm(dataset, desc="Checking valid data"):
        if 'labels' not in data or data['labels'] is None:
            continue
            
        original_labels = data['labels']
        noisy_labels = corrupt_process_labels(original_labels, rng)
        
        # Check if either version would pass max_length filter
        original_length = get_tokenized_length(data, tokenizer, original_labels)
        noisy_length = get_tokenized_length(data, tokenizer, noisy_labels)
        
        original_valid = max_length is None or original_length <= max_length
        noisy_valid = max_length is None or noisy_length <= max_length
        
        if original_valid or noisy_valid:
            valid_data.append({
                'data': data,
                'original_labels': original_labels,
                'noisy_labels': noisy_labels,
                'original_valid': original_valid,
                'noisy_valid': noisy_valid
            })
    
    # Step 2: Apply noise selection strategy
    # We want to end up with noise_ratio of the final dataset being noisy
    
    # Count how many examples can be both original and noisy
    both_valid = [x for x in valid_data if x['original_valid'] and x['noisy_valid']]
    only_original_valid = [x for x in valid_data if x['original_valid'] and not x['noisy_valid']]
    only_noisy_valid = [x for x in valid_data if not x['original_valid'] and x['noisy_valid']]
    
    # Strategy: 
    # 1. Include all only_noisy_valid as noisy
    # 2. Include all only_original_valid as original
    # 3. For both_valid, decide based on remaining noise ratio needed
    
    target_noisy_count = int((len(only_original_valid) + len(only_noisy_valid) + len(both_valid)) * noise_ratio)
    forced_noisy = len(only_noisy_valid)
    
    if forced_noisy >= target_noisy_count:
        # Too many forced noisy, randomly sample
        rng.shuffle(only_noisy_valid)
        final_noisy_from_forced = only_noisy_valid[:target_noisy_count]
        remaining_noisy_needed = 0
    else:
        final_noisy_from_forced = only_noisy_valid
        remaining_noisy_needed = target_noisy_count - forced_noisy
    
    # Handle both_valid based on remaining noise needed
    if remaining_noisy_needed > 0:
        rng.shuffle(both_valid)
        final_noisy_from_both = both_valid[:remaining_noisy_needed]
        final_original_from_both = both_valid[remaining_noisy_needed:]
    else:
        final_noisy_from_both = []
        final_original_from_both = both_valid
    
    # Step 3: Tokenize final dataset
    tokenized = []
    
    # Add original-only examples
    for item in tqdm(only_original_valid + final_original_from_both, desc="Tokenizing original"):
        result = tokenize_one_data(item['data'], tokenizer, mask_id, label_last_n, 
                                 max_length, orm, item['original_labels'])
        tokenized.extend(result)
    
    # Add noisy examples
    for item in tqdm(final_noisy_from_forced + final_noisy_from_both, desc="Tokenizing noisy"):
        result = tokenize_one_data(item['data'], tokenizer, mask_id, label_last_n, 
                                 max_length, orm, item['noisy_labels'])
        tokenized.extend(result)
    
    actual_noisy_count = len(final_noisy_from_forced) + len(final_noisy_from_both)
    total_count = len(tokenized)
    actual_noise_ratio = actual_noisy_count / total_count if total_count > 0 else 0
    
    print(f"Target noise ratio: {noise_ratio:.2%}")
    print(f"Actual noise ratio: {actual_noise_ratio:.2%} ({actual_noisy_count}/{total_count})")
    print(f"Forced noisy (only valid as noisy): {len(final_noisy_from_forced)}")
    print(f"Chosen noisy (could be either): {len(final_noisy_from_both)}")
    
    return tokenized

class TokenizedPRMDataset(Dataset):
    def __init__(self, data_path, category, tokenizer, label_mask_token_id=-100, 
                 label_last_n=None, max_length=None, orm=False, noise_ratio=0.0, seed=None):
        super().__init__()
        seed = 42 if seed is None else seed
        self.tokenized_data = tokenize_dataset(data_path, category, tokenizer, 
                                             label_mask_token_id, label_last_n, max_length, 
                                             orm, noise_ratio, seed)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, i):
        return self.tokenized_data[i]