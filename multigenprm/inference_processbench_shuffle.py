import argparse
import gc
import json
import time
import multiprocessing as mp
import os
import re
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from multigenprm.utils import split_dataset_for_gpus
from multigenprm.data.formats import CHAT_TEMPLATE, ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelWorker:
    """Worker class that maintains a persistent model instance."""
    
    def __init__(self, process_id, gpu_ids, args):
        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        self.process_id = process_id
        self.args = args
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        self.tokenizer.chat_template = CHAT_TEMPLATE
        
        # Initialize model
        print(f"Process {process_id} (GPUs {gpu_ids}): Initializing vLLM...")
        self.llm = LLM(
            model=args.model_id,
            tokenizer=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )
        print(f"Process {process_id}: Model ready")
        
        # Set up generation parameters
        self.sampling_params = SamplingParams(
            n=args.n_generation,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            stop=None
        )
        
        # Compile regex patterns
        if args.task_type == "orm":
            self.yes_no_pattern = re.compile(r'Verification: Is the answer correct \(Yes/No\)\? (Yes|No)')
        else:
            self.yes_no_pattern = re.compile(r"Is the solution correct\? (Yes|No)")
        self.prompt_format = ORM_PROMPT_FORMAT if args.task_type == "orm" else PRM_PROMPT_FORMAT
    
    def process_batch(self, split, dataset):
        # Prepare all prompts for the batch
        formatted_prompts = []
        for data in dataset:
            # prompt
            prompt = self.prompt_format(split, data["problem"], data["steps"])
            prompt = self.tokenizer.apply_chat_template(
                [{'role': "user", "content": prompt}], 
                tokenize=False, add_generation_prompt=True, add_special_tokens=False
            ) + "Let's verify step by step:"

            # vllm add bos token. So we should remove it.
            bos_token = self.tokenizer.bos_token
            if bos_token is not None and prompt.startswith(bos_token):
                prompt = prompt[len(bos_token):]
            
            formatted_prompts.append(prompt)
            
        # Generate
        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
            
        # Process outputs
        processed_results = []
        for output, item in zip(outputs, dataset):
            critiques, rewards = [], []
            for completion in output.outputs:
                # critique
                critiques.append(completion.text)
                
                # rewards
                match = self.yes_no_pattern.search(completion.text)
                if not match:
                    rewards.append(np.nan)
                else:
                    answer = match.group(1)
                    if answer == "Yes":
                        rewards.append(1.)
                    elif answer == "No":
                        rewards.append(0.)
                    else:
                        rewards.append(np.nan)
            
            result = {
                'id': item['id'],
                'problem': item['problem'],
                'steps': item['steps'],
                'critiques': critiques,
                'rewards': rewards,
                'label': item['label'],
                'final_answer_correct': item['final_answer_correct']
            }
            processed_results.append(result)
        
        return processed_results


def create_cache(args):
    """Initialize vLLM once to create cache, then delete it."""
    print("=" * 80)
    print("Creating vLLM kernel cache...")
    print(f"Model: {args.model_id}")
    print("=" * 80)
    
    # Create temporary LLM instance to build cache
    print("Initializing temporary vLLM instance (this may take a minute)...")
    
    temp_llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
        
    print("Cache created successfully. Cleaning up temporary model...")
    
    # Delete the model and free GPU memory
    del temp_llm
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Temporary model deleted. Cache is ready for use.")
    print("=" * 80)
    print()


def worker_process(process_id, gpu_ids, task_queue, result_queue, args):
    """Worker process that processes tasks from queue."""
    # Create worker with persistent model
    worker = ModelWorker(process_id, gpu_ids, args)
    
    # Process tasks from queue
    while True:
        task = task_queue.get()
        if task is None:  # Shutdown signal
            break
        
        split, dataset_chunk = task
        results = worker.process_batch(split, dataset_chunk)
        result_queue.put(results)
    
    print(f"Process {process_id}: Shutting down")


def main():
    parser = argparse.ArgumentParser(description="Generate text with VLLM and extract Yes/No probabilities")
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default='gsm8k', 
                       choices=['gsm8k', 'math', 'olympiadbench', 'omnimath', 'all'],
                       help="ProcessBench split to process")
    parser.add_argument("--task_type", type=str, default="orm", choices={"orm", "prm"})
    
    # Generation arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--n_generation", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--skip_cache", action="store_true", help="Skip cache creation (assume cache exists)")
    
    args = parser.parse_args()   
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    assert num_gpus % args.tensor_parallel_size == 0, \
        f"GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})"
    
    num_processes = num_gpus // args.tensor_parallel_size
    print(f"Using {num_gpus} GPUs with {num_processes} processes")
    
    # Determine splits to process
    if args.split == "all":
        splits = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    else:
        splits = [args.split]
    
    print(f"Will process splits: {splits}")
    
    # Create cache if not skipping
    if not args.skip_cache:
        create_cache(args)
        time.sleep(10)
    else:
        print("Skipping cache creation (assuming cache already exists)\n")
    
    # Single process mode
    if num_processes == 1:
        print("Using single process mode")
        gpu_ids = list(range(num_gpus))
        worker = ModelWorker(0, gpu_ids, args)
        
        for split in splits:
            print(f"\nProcessing {split}...")
            
            # Load dataset from Hugging Face
            dataset = load_dataset("Qwen/ProcessBench", split=split)
            dataset = [d for d in dataset]
            np.random.seed(args.seed)
            
            permuted_indices = np.random.permutation(len(dataset))
            permuted_dataset = []
            for index, data in enumerate(dataset):
                permuted_data = data.copy()
                permuted_steps = dataset[permuted_indices[index]]["steps"][:-1]
                permuted_steps.append(permuted_data["steps"][-1])
                permuted_data["steps"] = permuted_steps
                permuted_dataset.append(permuted_data)    
            
            dataset = permuted_dataset
            
            if args.debug:
                dataset = dataset[:2]
            
            print(f"Loaded {len(dataset)} items")
            
            results = worker.process_batch(split, dataset)
            
            # Save results
            output_file = os.path.join(args.output_dir, 
                f"debug_eval_{split}_dataset.json" if args.debug else f"eval_{split}_dataset.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            
            # Print statistics
            total = sum(len(r["rewards"]) for r in results)
            nan = sum(sum(np.isnan(reward) for reward in r["rewards"]) for r in results)
            print(f"Saved {len(results)} results to {output_file}")
            print(f"NaN/Total: {nan}/{total}")
    
    # Multi-process mode
    else:
        print(f"Using {num_processes} processes for parallel processing")
        mp.set_start_method('spawn', force=True)
        
        manager = mp.Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # Start worker processes
        processes = []
        for i in range(num_processes):
            gpu_ids = list(range(i * args.tensor_parallel_size, 
                                (i + 1) * args.tensor_parallel_size))
            p = mp.Process(target=worker_process, 
                          args=(i, gpu_ids, task_queue, result_queue, args))
            p.start()
            processes.append(p)
            print(f"Started worker process {i} with GPUs {gpu_ids}")
        
        print("\nWorkers initializing...")
        time.sleep(10)  # Give workers time to initialize
        print("Processing splits...\n")
        
        # Process each split
        for split in splits:
            print(f"Processing {split}...")
            
            # Load dataset from Hugging Face
            dataset = load_dataset("Qwen/ProcessBench", split=split)
            dataset = [d for d in dataset]
            np.random.seed(args.seed)
            
            permuted_indices = np.random.permutation(len(dataset))
            permuted_dataset = []
            for index, data in enumerate(dataset):
                permuted_data = data.copy()
                permuted_steps = dataset[permuted_indices[index]]["steps"][:-1]
                permuted_steps.append(permuted_data["steps"][-1])
                permuted_data["steps"] = permuted_steps
                permuted_dataset.append(permuted_data)    
            
            dataset = permuted_dataset
            
            if args.debug:
                dataset = dataset[:num_processes * 2]
            
            print(f"  Loaded {len(dataset)} items")
            
            # Split and distribute work
            chunks = split_dataset_for_gpus(dataset, num_processes)
            active_workers = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    task_queue.put((split, chunk))
                    active_workers += 1
                    print(f"  Assigned {len(chunk)} items to process {i}")
            
            # Collect results
            all_results = []
            for _ in range(active_workers):
                results = result_queue.get()
                all_results.extend(results)
            
            # Save results
            output_file = os.path.join(args.output_dir, 
                f"debug_eval_{split}_dataset.json" if args.debug else f"eval_{split}_dataset.json")
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=4)
            
            # Stats
            total = sum(len(r["rewards"]) for r in all_results)
            nan = sum(sum(np.isnan(reward) for reward in r["rewards"]) for r in all_results)
            print(f"  Saved {len(all_results)} results")
            print(f"  NaN/Total: {nan}/{total}")
            print(f"  Output: {output_file}\n")
        
        # Shutdown workers
        print("Shutting down workers...")
        for _ in range(num_processes):
            task_queue.put(None)
        for p in processes:
            p.join()
    
    print("\n" + "=" * 80)
    print("All processing complete!")
    processed_splits = []
    for split in splits:
        try:
            load_dataset("Qwen/ProcessBench", split=split)
            processed_splits.append(split)
        except:
            pass
    print(f"Splits processed: {len(processed_splits)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()