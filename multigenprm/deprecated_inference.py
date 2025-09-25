import argparse
import ast
import json
import math
import multiprocessing as mp
import os
import re
from multiprocessing import Manager

import numpy as np
import torch
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
    
    # Yes or No id
    yes_id = tokenizer.encode("Is the solution correct? Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode("Is the solution correct? No", add_special_tokens=False)[-1]
    
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
        logprobs=args.logprobs,
        stop=None
    )    
    
    # Description and check_input behavior
    desc = f"Process {process_id} processing"
    check_input = process_id == 0
    
    """Generate questions in batches"""
    # Process dataset in batches
    processed_results = []
    for batch_start in tqdm(range(0, len(dataset), args.batch_size), desc=desc):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch_data = dataset[batch_start:batch_end]
        
        # Prepare all prompts for the batch
        batch_formatted_prompts = []
        for data in batch_data:            
            for cot in data["cots"]:
                # prompt
                prompt = prompt_format(args.category, data["question"], cot)
                prompt = tokenizer.apply_chat_template(
                    [{'role': "user", "content": prompt}], 
                    tokenize=False, add_generation_prompt=True, add_special_tokens=False
                ) + "Let's verify step by step:"

                # vllm add bos token. So we should remove it.
                bos_token = tokenizer.bos_token
                if bos_token is not None and prompt.startswith(bos_token):
                    prompt = prompt[len(bos_token):]
                
                batch_formatted_prompts.append(prompt)
                
                if check_input:
                    print(prompt)
                    check_input = False

        # Generate for all prompts in the batch at once
        batch_outputs = llm.generate(batch_formatted_prompts, sampling_params, use_tqdm=False)

        # Process outputs and group them back by question
        batch_results, output_idx = {}, 0
        for q_idx, data in enumerate(batch_data):
            batch_results[q_idx] = {
                "data": data,
                "critiques": [],
                "logprobs": [],
                "rewards": []
            }
            
            for _ in range(len(data["cots"])):
                batch_output = batch_outputs[output_idx]
                
                # Extract critiques and rewards for this output
                critiques, logprobs, rewards = [], [], []
                for completion in batch_output.outputs:
                    # critique
                    critiques.append(completion.text)
                    
                    # logprobs
                    lp_dict = completion.logprobs[-2]
                    lp_dict = {key: float(lp_dict[key].logprob) for key in lp_dict}
                    logprobs.append(lp_dict)
                    
                    # rewards
                    match = yes_no_pattern.search(completion.text)
                    if not match:
                        rewards.append(np.nan)
                        continue
                    
                    # Get Yes and No logprobs
                    yes_lp = lp_dict.get(yes_id, None)
                    no_lp = lp_dict.get(no_id, None)
                    
                    if yes_lp is not None and no_lp is not None:
                        exp_yes = math.exp(yes_lp / args.decision_temperature)
                        exp_no = math.exp(no_lp / args.decision_temperature)
                        rewards.append(exp_yes / (exp_yes + exp_no))
                    elif yes_lp is not None:
                        rewards.append(1.0)
                    elif no_lp is not None:
                        rewards.append(0.0)
                    else:
                        rewards.append(np.nan)
                
                batch_results[q_idx]["critiques"].append(critiques)
                batch_results[q_idx]["logprobs"].append(logprobs)
                batch_results[q_idx]["rewards"].append(rewards)
                
                output_idx += 1
        
        # Convert batch results to the expected format
        for q_idx in sorted(batch_results.keys()):
            result = batch_results[q_idx]
            result_data = {
                "q_id": result["data"]["q_id"],
                "cot_ids": result["data"]["cot_ids"],
                "answer": result["data"]["answer"],
                "parsed_answers": result["data"]["parsed_answers"],
                "critiques": result["critiques"],
                "logprobs": result["logprobs"],
                "rewards": result["rewards"],
            }
            processed_results.append(result_data)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(processed_results, f, indent=4)
    else:
        return processed_results


def main():
    parser = argparse.ArgumentParser(description="Generate text with VLLM and extract Yes/No probabilities")
    
    # I/O arguments
    parser.add_argument("--input_dir", type=str, default="./local_datasets/test", help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other'])
    parser.add_argument("--task_type", type=str, default="orm", choices={"orm", "prm"})
    
    # Generation arguments
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--n_generation", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    
    # Batch processing argument
    parser.add_argument("--batch_size", type=int, default=1, help="Number of questions to process in each batch")
    
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
    
    # Load dataset
    print(f"Loading dataset for category: {args.category}...")
    with open(os.path.join(args.input_dir, f"{args.category}_dataset.json"), "r") as f:
        dataset = json.load(f)
    
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
                temp_file = os.path.join(args.output_dir, f"{args.category}_temp_file_{process_id}.json")
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
    
    # Check
    total, nan = 0, 0        
    for results in output_results:
        for cot_rewards in results["rewards"]:
            total += len(cot_rewards)
            nan += sum([int(np.isnan(r)) for r in cot_rewards])
    print(f"# of Nan / Total: {total} / {nan}")
        
    output_file = os.path.join(args.output_dir, 
        "debug_eval_dataset.json" if args.debug else f"eval_{args.category}_dataset.json")
    with open(output_file, "w") as f:
        json.dump(output_results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    if num_processes > 1:
        print(f"Processed {len(output_results)} items using {num_processes} processes with {args.tensor_parallel_size} GPUs each")
    else:
        print(f"Processed {len(output_results)} items using single process with {num_gpus} GPUs")

if __name__ == "__main__":
    main()