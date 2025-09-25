import argparse
import json
import math
import multiprocessing as mp
import os
import re
from multiprocessing import Manager

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from multigenprm.utils import split_dataset_for_gpus
from multigenprm.data.formats import CHAT_TEMPLATE, ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_gpu_batch(process_id, gpu_ids, dataset, args, temp_file=None, lock=None):
    if temp_file is not None:
        # Set CUDA_VISIBLE_DEVICES to the specific GPUs for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
    # prompt_format
    prompt_format = ORM_PROMPT_FORMAT if args.task_type == "orm" else PRM_PROMPT_FORMAT
    
    # Set variables
    yes_no_pattern = re.compile(r"Is the solution correct\? (Yes|No)")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.chat_template = CHAT_TEMPLATE
    
    if lock:
        print(f"Process {process_id} (GPUs {gpu_ids}): Waiting to acquire lock for model initialization...")
        lock.acquire()
        print(f"Process {process_id} (GPUs {gpu_ids}): Lock acquired. Initializing vLLM...")   
    
    # Initialize VLLM
    llm = LLM(
        model=args.model_id, 
        tokenizer=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    
    # Release the lock as soon as initialization is done so the next process can start.
    if lock:
        lock.release()
        print(f"Process {process_id} (GPUs {gpu_ids}): vLLM initialized and lock released. Starting inference.")    
        
    # Sampling params
    sampling_params = SamplingParams(
        n=args.n_generation,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        stop=None
    )
    
    # Description and check_input behavior
    check_input = process_id == 0
    
    """Generate questions in batches"""        
    # Prepare all prompts for the batch
    formatted_prompts = []
    for data in dataset:
        # Get split name from the data (assuming it's stored in the combined dataset)
        split_name = data.get('split', args.split)
        
        # prompt
        prompt = prompt_format(split_name, data["problem"], data["steps"])
        prompt = tokenizer.apply_chat_template(
            [{'role': "user", "content": prompt}], 
            tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ) + "Let's verify step by step:"

        # vllm add bos token. So we should remove it.
        bos_token = tokenizer.bos_token
        if bos_token is not None and prompt.startswith(bos_token):
            prompt = prompt[len(bos_token):]
        
        formatted_prompts.append(prompt)
        
        if check_input:
            print(prompt)
            check_input = False

    # Generate for all prompts in the batch at once
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)

    # Process outputs and group them back by question
    processed_results = []
    for output, item in zip(outputs, dataset):
        critiques, rewards = [], []
        for completion in output.outputs:
            # critique
            critiques.append(completion.text)
            
            # rewards
            match = yes_no_pattern.search(completion.text)
            if not match:
                rewards.append(np.nan)
                continue
            
            if "Is the solution correct? Yes" in completion.text:
                rewards.append(1.)
            elif "Is the solution correct? No" in completion.text:
                rewards.append(0.)
            else:
                rewards.append(np.nan)
            
        result = {
            'id': item['id'],
            'problem': item['problem'],
            'steps': item['steps'],
            'critiques': critiques,
            'rewards': rewards,
            'final_answer_correct': item['final_answer_correct'],
            'split': item.get('split', args.split)  # Store split info
        }
        processed_results.append(result)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(processed_results, f, indent=4)
    else:
        return processed_results


def main():
    parser = argparse.ArgumentParser(description="Generate text with VLLM and extract Yes/No probabilities")
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default='gsm8k', 
                       choices=['gsm8k', 'math', 'olympiadbench', 'omnimath', 'all'],
                       help="ProcessBench split to process (use 'all' for all splits)")
    parser.add_argument("--task_type", type=str, default="orm", choices={"orm", "prm"})
    
    # Generation arguments
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--n_generation", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()   
    
    # Make dirs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU configuration
    num_gpus = torch.cuda.device_count()
    assert num_gpus % args.tensor_parallel_size == 0, f"Number of GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})"
    
    num_processes = num_gpus // args.tensor_parallel_size
    print(f"Using {num_gpus} GPUs with tensor_parallel_size={args.tensor_parallel_size}, creating {num_processes} processes")
    
    # Only set multiprocessing start method if using multiple processes
    if num_processes > 1:
        mp.set_start_method('spawn', force=True)
    
    # Load dataset(s)
    if args.split == 'all':
        print("\nLoading all ProcessBench splits...")
        all_splits = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
        dataset = []
        for split_name in all_splits:
            print(f"  Loading split: {split_name}...")
            split_data = load_dataset("Qwen/ProcessBench", split=split_name)
            split_data = [dict(d, split=split_name) for d in split_data]  # Add split info to each item
            dataset.extend(split_data)
            print(f"    Loaded {len(split_data)} items from {split_name}")
        print(f"Total items loaded: {len(dataset)}")
    else:
        print(f"\nLoading ProcessBench dataset, split: {args.split}...")
        dataset = load_dataset("Qwen/ProcessBench", split=args.split)
        dataset = [dict(d, split=args.split) for d in dataset]
        print(f"Loaded {len(dataset)} items")
    
    if args.debug:
        dataset = dataset[:num_processes]
    
    # Use single process if only 1 GPU (regardless of how we got to 1)
    if num_processes > 1:
        print(f"Using {num_processes} processes for processing")
        
        # Split dataset for multiple processes
        dataset_batches = split_dataset_for_gpus(dataset, num_processes)
        
        # Create Manager
        manager = Manager()
        lock = manager.Lock()
        
        # Create processes for each GPU
        processes, temp_file_list = [], []
        for process_id in range(num_processes):
            if len(dataset_batches[process_id]) > 0:  # Only spawn process if there's data
                # temp_file
                temp_file = os.path.join(args.output_dir, f"temp_file_{process_id}.json")
                temp_file_list.append(temp_file)
                
                # Calculate GPU IDs for this process
                start_gpu = process_id * args.tensor_parallel_size
                gpu_ids = list(range(start_gpu, start_gpu + args.tensor_parallel_size))
                
                p = mp.Process(
                    target=process_gpu_batch,
                    args=(process_id, gpu_ids, dataset_batches[process_id], args, temp_file, lock)
                )
                processes.append(p)
                p.start()
                print(f"Started process {process_id} with GPUs {gpu_ids}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
        # Merge results
        output_results = []
        for temp_file in temp_file_list:
            with open(temp_file, "r") as f:
                output_results.extend(json.load(f))
            os.remove(temp_file)    
        
    else:
        print("Using single process processing")
        # For single process, use all available GPUs
        gpu_ids = list(range(num_gpus))
        output_results = process_gpu_batch(0, gpu_ids, dataset, args, None, None)
    
    # Save results
    if args.split == 'all':
        # Separate results by split and save individually
        results_by_split = {}
        for result in output_results:
            split_name = result['split']
            if split_name not in results_by_split:
                results_by_split[split_name] = []
            # Remove the split field from individual results as it's redundant in split-specific files
            result_copy = {k: v for k, v in result.items() if k != 'split'}
            results_by_split[split_name].append(result_copy)
        
        # Save each split's results
        for split_name, split_results in results_by_split.items():
            output_file = os.path.join(args.output_dir, 
                f"debug_eval_{split_name}_dataset.json" if args.debug else f"eval_{split_name}_dataset.json")
            with open(output_file, "w") as f:
                json.dump(split_results, f, indent=4)
            print(f"Results for {split_name} saved to {output_file} ({len(split_results)} items)")
        
        print(f"\nTotal results saved: {len(output_results)} items across {len(results_by_split)} splits")
    else:
        # Save single split results
        output_file = os.path.join(args.output_dir, 
            "debug_eval_dataset.json" if args.debug else f"eval_{args.split}_dataset.json")
        # Remove split field from results
        clean_results = [{k: v for k, v in r.items() if k != 'split'} for r in output_results]
        with open(output_file, "w") as f:
            json.dump(clean_results, f, indent=4)
        print(f"Results saved to {output_file}")
    
    if num_processes > 1:
        print(f"Processed {len(output_results)} items using {num_processes} processes with {args.tensor_parallel_size} GPUs each")
    else:
        print(f"Processed {len(output_results)} items using single process with {num_gpus} GPUs")


if __name__ == "__main__":
    main()