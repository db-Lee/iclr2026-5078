import json, os
import argparse
from tqdm import tqdm
import torch
import multiprocessing as mp
from datasets import load_dataset

import numpy as np
from multigenprm.utils import split_dataset_for_gpus
from versaprm.prm import PRM

def prepare_text_for_prm(item):
    """Prepare text from ProcessBench item for PRM evaluation - following exact format"""
    # Strip and clean the problem and steps
    problem = item['problem'].strip().replace(' \n\n\n\n', '')
    steps = [step.strip().replace(' \n\n\n\n', '') for step in item['steps']]
    
    # Add the specific formatting with \n\n\n\n
    updated_steps = [f'{step} \n\n\n\n' for step in steps]
    steps_all = f'{problem} \n\n' + ''.join(updated_steps)
    
    return steps_all

def process_gpu_batch(gpu_id, dataset, args, temp_file=None):
    """Process a batch of data on specific GPU"""
    print(f"Process {gpu_id}: Initializing PRM on GPU {gpu_id}...")
    
    # Initialize PRM model with specific GPU device
    prm = PRM(model_id=args.model_id, aggregation="full", device=torch.device(f'cuda:{gpu_id}'))
    
    print(f"Process {gpu_id}: PRM initialized. Starting processing...")
    print(f"Process {gpu_id}: Total items to process: {len(dataset)}")
    
    # Process the data
    reward_results = []
    for i in tqdm(range(0, len(dataset), args.batch_size), desc=f'Process {gpu_id}: Processing batches'):
        batch = dataset[i:i + args.batch_size]
        
        # Prepare texts for PRM
        batch_texts = [prepare_text_for_prm(item) for item in batch]
        
        # Get rewards for this batch
        batch_rewards = prm(batch_texts)
        
        # Store rewards with original data
        for rewards, item in zip(batch_rewards, batch):
            result = {
                'id': item['id'],
                'problem': item['problem'],
                'steps': item['steps'],
                'rewards': rewards,
                'final_answer_correct': item['final_answer_correct']
            }
            reward_results.append(result)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(reward_results, f, indent=4)
        print(f"Process {gpu_id}: Results saved to {temp_file}")
    else:
        return reward_results

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Process rewards for PRM models on ProcessBench')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_id', type=str, required=True, help="PRM model ID")
    parser.add_argument('--output_dir', type=str, required=True, 
                       help="Directory to save results")
    parser.add_argument("--split", type=str, default='gsm8k', 
                       choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'],
                       help="ProcessBench split to process")
    parser.add_argument('--batch_size', type=int, default=8, 
                       help="Batch size for processing")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU configuration
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs, creating {num_gpus} processes (1 GPU per process)")
    
    # Only set multiprocessing start method if using multiple processes
    if num_gpus > 1:
        mp.set_start_method('spawn', force=True)

    print(f"\nLoading ProcessBench dataset, split: {args.split}...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("Qwen/ProcessBench", split=args.split)
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
        dataset = dataset[:num_gpus]
    
    print(f"Dataset loaded: {len(dataset)} items")
    
    # Use single process if only 1 GPU available
    if num_gpus > 1:
        print(f"Using {num_gpus} processes for processing")
        
        # Split dataset for multiple processes
        dataset_batches = split_dataset_for_gpus(dataset, num_gpus)
        
        # Create processes for each GPU
        processes, temp_file_list = [], []
        for gpu_id in range(num_gpus):
            if len(dataset_batches[gpu_id]) > 0:  # Only spawn process if there's data
                # temp_file
                temp_file = os.path.join(args.output_dir, f"{args.split}_temp_file_{gpu_id}.json")
                temp_file_list.append(temp_file)
                
                p = mp.Process(
                    target=process_gpu_batch,
                    args=(gpu_id, dataset_batches[gpu_id], args, temp_file)
                )
                processes.append(p)
                p.start()
                print(f"Started process on GPU {gpu_id}")
        
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
        # For single process, use GPU 0
        # Convert dataset to list for processing
        dataset_list = list(dataset)
        output_results = process_gpu_batch(0, dataset_list, args, None)

    # Save final results
    output_file = os.path.join(args.output_dir, 
        f"debug_{args.split}_dataset.json" if args.debug else f"eval_{args.split}_dataset.json")
    with open(output_file, "w") as f:
        json.dump(output_results, f, indent=4)
        
    print(f"\nResults for {args.split} saved to {output_file}")
    print(f"Processed {len(output_results)} items")

if __name__ == '__main__':
    main()