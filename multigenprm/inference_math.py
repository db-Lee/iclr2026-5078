import os
import re
import gc
import json
import math
import time
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing as mp

import torch
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
        
        # Get Yes/No token IDs
        self.yes_id = self.tokenizer.encode(" Yes", add_special_tokens=False)[-1]
        self.no_id = self.tokenizer.encode(" No", add_special_tokens=False)[-1]
        
        # Initialize model
        print(f"Process {process_id} (GPUs {gpu_ids}): Initializing vLLM...")
        
        # Define stop sequences based on task type - stop at the exact verdict phrases
        if args.task_type == "orm":
            stop_sequences = [
                "Verification: Is the answer correct (Yes/No)? Yes",
                "Verification: Is the answer correct (Yes/No)? No"
            ]
            self.yes_no_pattern = re.compile(r'Verification: Is the answer correct \(Yes/No\)\? (Yes|No)')
        else:
            stop_sequences = [
                "Is the solution correct? Yes",
                "Is the solution correct? No"
            ]
            self.yes_no_pattern = re.compile(r"Is the solution correct\? (Yes|No)")
        
        self.llm = LLM(
            model=args.model_id,
            tokenizer=args.model_id,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )
        print(f"Process {process_id}: Model ready")
        
        # Set up generation parameters with stop sequences
        self.sampling_params = SamplingParams(
            n=args.n_generation,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            logprobs=args.logprobs,
            stop=stop_sequences,  # This will stop generation after Yes or No
            include_stop_str_in_output=True  # Include the stop token in output
        )
        
        self.prompt_format = ORM_PROMPT_FORMAT if args.task_type == "orm" else PRM_PROMPT_FORMAT
    
    def process_batch(self, category, dataset):
        """Process a dataset batch for a specific category."""
        processed_results = []
        
        for batch_start in tqdm(range(0, len(dataset), self.args.batch_size), 
                               desc=f"P{self.process_id}-{category}"):
            batch_end = min(batch_start + self.args.batch_size, len(dataset))
            batch_data = dataset[batch_start:batch_end]
            
            # Prepare prompts
            batch_prompts = []
            for data in batch_data:
                for cot in data["cots"]:
                    if self.args.task_type == "prm":
                        prompt = self.prompt_format(category, data["question"], cot, False)
                    else:
                        prompt = self.prompt_format(category, data["question"], cot)
                    prompt = self.tokenizer.apply_chat_template(
                        [{'role': "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True, add_special_tokens=False
                    ) + "Let's verify step by step:"
                    
                    # Remove BOS token if present (vLLM adds it)
                    if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
                        prompt = prompt[len(self.tokenizer.bos_token):]
                    
                    batch_prompts.append(prompt)
            
            # Generate
            outputs = self.llm.generate(batch_prompts, self.sampling_params, use_tqdm=False)
            
            # Process outputs
            output_idx = 0
            for data in batch_data:
                result = {
                    "q_id": data["q_id"],
                    "cot_ids": data["cot_ids"],
                    "answer": data["answer"],
                    "parsed_answers": data["parsed_answers"],
                    "critiques": [],
                    "logprobs": [],
                    "rewards": []
                }
                
                for _ in data["cots"]:
                    critiques, logprobs, rewards = [], [], []
                    
                    for completion in outputs[output_idx].outputs:
                        # Critique text
                        critiques.append(completion.text)
                        
                        # Get logprobs for the Yes/No token
                        # With stop sequences and include_stop_str_in_output=True, 
                        # the Yes/No token should be at [-1] position
                        if completion.logprobs and len(completion.logprobs) > 0:
                            # Use [-1] since stop sequence tokens are included
                            lp_dict = {k: float(v.logprob) for k, v in completion.logprobs[-1].items()}
                            logprobs.append(lp_dict)
                            
                            # Calculate reward
                            yes_lp = lp_dict.get(self.yes_id)
                            no_lp = lp_dict.get(self.no_id)
                            
                            if yes_lp is not None and no_lp is not None:
                                exp_yes = math.exp(yes_lp / self.args.decision_temperature)
                                exp_no = math.exp(no_lp / self.args.decision_temperature)
                                rewards.append(exp_yes / (exp_yes + exp_no))
                            elif yes_lp is not None and no_lp is None:
                                rewards.append(1.0)
                            elif yes_lp is None and no_lp is not None:
                                rewards.append(0.0)
                            else:
                                rewards.append(np.nan)
                        else:
                            logprobs.append({})
                            rewards.append(np.nan)
                    
                    result["critiques"].append(critiques)
                    result["logprobs"].append(logprobs)
                    result["rewards"].append(rewards)
                    output_idx += 1
                
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
        
        category, dataset_chunk = task
        results = worker.process_batch(category, dataset_chunk)
        result_queue.put(results)
    
    print(f"Process {process_id}: Shutting down")


def main():
    parser = argparse.ArgumentParser(description="Generate text with VLLM and extract Yes/No probabilities")
    
    # I/O arguments
    parser.add_argument("--input_dir", type=str, default="./local_datasets/test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--category", type=str, default="all", 
                       choices=['gsm8k', 'math', 'all'])
    parser.add_argument("--task_type", type=str, default="orm", choices=["orm", "prm"])
    
    # Model arguments
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--n_generation", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_cache", action="store_true", help="Skip cache creation (assume cache exists)")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    assert num_gpus % args.tensor_parallel_size == 0, \
        f"GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})"
    
    num_processes = num_gpus // args.tensor_parallel_size
    print(f"Using {num_gpus} GPUs with {num_processes} processes")
    
    # Determine categories to process
    if args.category == "all":
        categories = ['gsm8k', 'math']
    else:
        categories = [args.category]
    
    print(f"Will process categories: {categories}")
    
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
        
        for category in categories:
            dataset_file = os.path.join(args.input_dir, f"{category}_dataset.json")
            if not os.path.exists(dataset_file):
                print(f"Skipping {category}: file not found")
                continue
            
            print(f"\nProcessing {category}...")
            with open(dataset_file, "r") as f:
                dataset = json.load(f)
            
            if args.debug:
                dataset = dataset[:2]
            
            results = worker.process_batch(category, dataset)
            
            # Save results
            output_file = os.path.join(args.output_dir, 
                f"debug_eval_{category}_dataset.json" if args.debug else f"eval_{category}_dataset.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            
            # Print statistics
            total = sum(len(r["rewards"]) * len(r["rewards"][0]) for r in results if r["rewards"])
            nan = sum(sum(np.isnan(reward).sum() for reward in r["rewards"]) for r in results)
            print(f"Saved {len(results)} results to {output_file}")
            print(f"NaN/Total rewards: {nan}/{total}")
    
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
        print("Processing categories...\n")
        
        # Process each category
        for category in categories:
            dataset_file = os.path.join(args.input_dir, f"{category}_dataset.json")
            if not os.path.exists(dataset_file):
                print(f"Skipping {category}: file not found")
                continue
            
            print(f"Processing {category}...")
            with open(dataset_file, "r") as f:
                dataset = json.load(f)
            
            if args.debug:
                dataset = dataset[:num_processes * 2]
            
            print(f"  Loaded {len(dataset)} items")
            
            # Split and distribute work
            chunks = split_dataset_for_gpus(dataset, num_processes)
            active_workers = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    task_queue.put((category, chunk))
                    active_workers += 1
                    print(f"  Assigned {len(chunk)} items to process {i}")
            
            # Collect results
            all_results = []
            for _ in range(active_workers):
                results = result_queue.get()
                all_results.extend(results)
            
            # Save results
            output_file = os.path.join(args.output_dir, 
                f"debug_eval_{category}_dataset.json" if args.debug else f"eval_{category}_dataset.json")
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=4)
            
            # Stats
            total = sum(len(r["rewards"]) * len(r["rewards"][0]) for r in all_results if r["rewards"])
            nan = sum(sum(np.isnan(reward).sum() for reward in r["rewards"]) for r in all_results)
            print(f"  Saved {len(all_results)} results")
            print(f"  NaN/Total rewards: {nan}/{total}")
            print(f"  Output: {output_file}\n")
        
        # Shutdown workers
        print("Shutting down workers...")
        for _ in range(num_processes):
            task_queue.put(None)
        for p in processes:
            p.join()
    
    print("\n" + "=" * 80)
    print("All processing complete!")
    print(f"Categories processed: {len([c for c in categories if os.path.exists(os.path.join(args.input_dir, f'{c}_dataset.json'))])}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()