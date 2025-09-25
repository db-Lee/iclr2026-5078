import argparse
import json
import multiprocessing as mp
import os
from multiprocessing import Manager

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from multigenprm.utils import split_dataset_for_gpus
from multigenprm.data.formats import ORM_PROMPT_FORMAT, DATA_PRM_PROMPT_FORMAT

def process_gpu_batch(process_id, gpu_ids, dataset, args, temp_file=None, lock=None):
    if temp_file is not None:
        # Set CUDA_VISIBLE_DEVICES to the specific GPUs for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
    # prompt_format
    prompt_format = ORM_PROMPT_FORMAT if args.task_type == "orm" else DATA_PRM_PROMPT_FORMAT
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # if lock:
    #     print(f"Process {process_id} (GPUs {gpu_ids}): Waiting to acquire lock for model initialization...")
    #     lock.acquire()
    #     print(f"Process {process_id} (GPUs {gpu_ids}): Lock acquired. Initializing vLLM...")   
    
    # Initialize vLLM with tensor parallelism
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    
    # Release the lock as soon as initialization is done so the next process can start.
    # if lock:
    #     lock.release()
    #     print(f"Process {process_id} (GPUs {gpu_ids}): vLLM initialized and lock released. Starting inference.")    
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        n=args.n_generation,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        stop=None
    )
    
    # Check_input behavior
    check_input = process_id == 0
    
    """Generate question one by one"""
    formatted_prompts, meta_data = [], []
    for data in dataset:
        question, cot, labels = data["question"], data["cot"], data["labels"]
        
        if -1 in labels:
            length = labels.index(-1)
            if length > 0:
                truncated_cot, truncated_labels = cot[:length], labels[:length]

                # prompt
                prompt = prompt_format(args.category, question, truncated_cot)
                prompt = tokenizer.apply_chat_template(
                    [{'role': "user", "content": prompt}], 
                    tokenize=False, add_generation_prompt=True, add_special_tokens=False
                ) + "Let's verify step by step:"
                
                # vllm add bos token. So we should remove it.
                bos_token = tokenizer.bos_token
                if bos_token is not None and prompt.startswith(bos_token):
                    prompt = prompt[len(bos_token):]
                    
                formatted_prompts.append(prompt)
                
                # append meta_data
                _data = data.copy()
                _data["cot"] = truncated_cot
                _data["labels"] = truncated_labels
                meta_data.append(_data)
                
                if check_input:
                    print(prompt)
                    check_input = False

    assert len(formatted_prompts) == len(meta_data)
    llm_outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)

    # Extract critiques
    processed_results = []
    for idx, llm_output in enumerate(llm_outputs):        
        for completion in llm_output.outputs:
            result_data = meta_data[idx].copy()
            result_data["critique"] = completion.text
            processed_results.append(result_data)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(processed_results, f, indent=4)
    else:
        return processed_results

def main():    
    parser = argparse.ArgumentParser(description="Generate step-by-step verification critique data for MMLU-Pro-CoT")
    
    # I/O arguments
    parser.add_argument("--input_dir", type=str, default="local_datasets/train", help="Directory to load data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--category", type=str, default="law", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', 'prm800k'
    }, help="Category of problems to process")
    parser.add_argument("--task_type", type=str, default="orm", choices={"orm", "prm"})
    
    # Generation arguments
    parser.add_argument("--model_id", type=str, default="Qwen/QwQ-32B", help="Model to use for generation")
    parser.add_argument("--n_generation", type=int, default=4, help="Number of generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max number of tokens")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top P")
    parser.add_argument("--top_k", type=int, default=20, help="Top K")
    parser.add_argument("--min_p", type=float, default=0., help="Min P")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for each vLLM instance")
    
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    # Make dirs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU configuration
    num_gpus = torch.cuda.device_count()
    assert num_gpus % args.tensor_parallel_size == 0, \
        f"Number of GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})"
    
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
    
    # Use single process if only 1 process needed
    if num_processes > 1:
        print(f"Using {num_processes} processes for processing")
        
        # Split dataset for multiple processes
        dataset_batches = split_dataset_for_gpus(dataset, num_processes)
        
        # Create Manager
        manager = Manager()
        lock = None # manager.Lock()
        
        # Create processes for each GPU group
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
    
    output_file = os.path.join(args.output_dir, 
        "debug_dataset.json" if args.debug else f"{args.category}_dataset.json")
    with open(output_file, "w") as f:
        json.dump(output_results, f, indent=4)
        
    print(f"Results saved to {output_file}")
    if num_processes > 1:
        print(f"Processed {len(output_results)} items using {num_processes} processes with {args.tensor_parallel_size} GPUs each")
    else:
        print(f"Processed {len(output_results)} items using single process with {num_gpus} GPUs")
    
if __name__ == "__main__":
    main()