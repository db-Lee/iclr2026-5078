import re
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import is_conversational
from datasets import Dataset, concatenate_datasets

from multigenprm.data.formats import ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT

def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
    # tokenize function from SFTTrainer._prepare_dataset()
    if "prompt" in example:  # prompt-completion case
        if is_conversational(example):
            prompt_ids = processing_class.apply_chat_template(
                example["prompt"],
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            prompt_completion_ids = processing_class.apply_chat_template(
                example["prompt"] + example["completion"],
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
        else:
            prompt_ids = processing_class(text=example["prompt"]).input_ids
            prompt_completion_ids = processing_class(
                text=example["prompt"] + example["completion"]
            ).input_ids

        # Check if the tokenized prompt starts with the tokenized prompt+completion
        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
            warnings.warn(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                "token handling. Verify that the tokenizer is processing text consistently."
            )

        # Create a completion mask
        completion_mask = [0] * len(prompt_ids) + [1] * \
            (len(prompt_completion_ids) - len(prompt_ids))
        processed = {"input_ids": prompt_completion_ids,
                     "completion_mask": completion_mask}
    else:  # language modeling case
        if is_conversational(example):
            processed = processing_class.apply_chat_template(
                example["messages"],
                return_dict=True,
                return_assistant_tokens_mask=assistant_only_loss,
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                raise RuntimeError(
                    "You're using `assistant_only_loss=True`, but at least one example has no "
                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                    "check the template and ensure it's correctly configured to support assistant "
                    "masking."
                )
            processed = {k: processed[k] for k in (
                "input_ids", "assistant_masks") if k in processed}
        else:
            processed = {"input_ids": processing_class(
                text=example[dataset_text_field]).input_ids}
    return processed

def add_eos(example, eos_token):
    # language modeling case
    if "text" in example and not example["text"].endswith(eos_token):
        example["text"] = example["text"] + eos_token
    elif "completion" in example and not example["completion"].endswith(eos_token):
        example["completion"] = example["completion"] + eos_token
    return example

def preprocess_dataset(dataset, tokenizer):
     # add eos token
    map_kwargs = {}
    map_kwargs["desc"] = f"Adding EOS to dataset"
    dataset = dataset.map(add_eos,
                fn_kwargs={"eos_token": tokenizer.eos_token},
                remove_columns=None,
                **map_kwargs,
                )
    # tokenize dataset
    map_kwargs["desc"] = "Tokenize train dataset"
    dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "processing_class": tokenizer,
            "dataset_text_field": "text",
            "assistant_only_loss": False,
        },
        **map_kwargs
    )
    return dataset

def load_train_dataset(task_type, data_path, category):
    prompt_format = ORM_PROMPT_FORMAT if task_type == "orm" else PRM_PROMPT_FORMAT
    def _load_dataset(_category):
        with open(os.path.join(data_path, f"preprocessed_{_category}_dataset.json"), "r") as f:
            dataset = json.load(f)
        
        formatted_dataset = []
        for data in tqdm(dataset, desc=f"Processing {_category}"):
            prompt = prompt_format(_category, data["question"], data["cot"])
            formatted_dataset.append({
                "prompt": f"<｜User｜>{prompt}",
                "completion": f"<｜Assistant｜>{data['critique']}"
            })           
    
        return Dataset.from_list(formatted_dataset)
    
    if category == "all":
        categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 'history', 
                     'economics', 'math', 'business', 'philosophy', 'health', 'engineering', 
                     'computer_science', 'other']
        dataset = concatenate_datasets([
            _load_dataset(category) for category in categories
        ])
    else:
        dataset = _load_dataset(category)
        
    return dataset

def parse_orm_label(text):
    """Parses the final Yes/No verdict from the critique text."""
    pattern = r'Verification: Is the answer correct \(Yes/No\)\?\s*(Yes|No)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        verdict = match.group(1).lower()
        if verdict == 'yes':
            return 1
        elif verdict == 'no':
            return -1
        else:
            return np.nan
    return np.nan

def parse_prm_label(text):
    """
    Parses model output and returns an array of 1 (correct) or -1 (incorrect)
    for exact phrases like 'The step is \\boxed{correct}' or 'The step is \\boxed{incorrect}'.
    """
    # Match literal: The step is \\boxed{correct}
    pattern = r'The step is \\boxed{(correct|incorrect)}'
    verdicts = re.findall(pattern, text, re.IGNORECASE)
    array = []
    for v in verdicts:
        if v.lower() == 'correct':
            array.append(1)
        elif v.lower() == 'incorrect':
            array.append(-1)
        else:
            array.append(np.nan)
    return array

def trim_after_first_verdict(text):
    """Trims the text after the first Yes/No verdict."""
    # Fixed pattern - removed space before (Yes/No)
    pattern = r'Verification: Is the answer correct \(Yes/No\)\?\s*(Yes|No)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return text[:match.end()]
    else:
        return text  # or "" or None if you prefer to indicate no match

def truncate_after_last_boxed_step(text):
    """
    Truncates the text after the last occurrence of:
    'The step is \\boxed{correct}' or 'The step is \\boxed{incorrect}'.
    Keeps the matching line.
    """
    pattern = r'The step is \\boxed{(correct|incorrect)}'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return text  # nothing to truncate

    last_match = matches[-1]
    end_index = last_match.end()  # keep the full match
    return text[:end_index]

def split_dataset_for_gpus(dataset, num_gpus):
    """Split dataset list into batches for each GPU"""
    batch_size = len(dataset) // num_gpus
    
    batches = []
    for i in range(num_gpus):
        start_idx = i * batch_size
        if i == num_gpus - 1:  # Last GPU gets remaining items
            end_idx = len(dataset)
        else:
            end_idx = (i + 1) * batch_size
        
        batch_data = dataset[start_idx:end_idx]
        batches.append(batch_data)
    
    return batches

def merge_adapter_and_save_temp(checkpoint_path, output_dir):
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("Adapter found, merging with base model...")
        
        temp_dir = os.path.join(output_dir, "tmp")
        if os.path.exists(temp_dir):
            print("Merged model found, end merging.")
            return
    
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError("base_model_name_or_path not found in adapter_config.json")
        
        # Load and merge
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="cpu", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_model = model.merge_and_unload()

        # Tokenzier
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)    
        
        # Save to temp directory
        merged_model.save_pretrained(temp_dir, safe_serialization=True)    
        tokenizer.save_pretrained(temp_dir)
        print("Model merging completed")
    else:
        print("No adapter found, using checkpoint directly")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.checkpoint_path
    merge_adapter_and_save_temp(args.checkpoint_path, args.output_dir)

if __name__ == "__main__":
    main()