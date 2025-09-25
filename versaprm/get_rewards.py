import json, os
import argparse
from tqdm import tqdm
import torch
import multiprocessing as mp
from multiprocessing import Manager

from multigenprm.utils import split_dataset_for_gpus
from versaprm.prm import PRM

def flatten_all_data(dataset):
    """Flatten all CoTs from entire dataset into single list with tracking info"""
    flattened = []
    
    for q_idx, data in enumerate(dataset):
        for cot_idx, cot in enumerate(data['cots']):
            steps = [step.strip().replace(' \n\n\n\n', '') for step in cot]
            question = data['question'].strip().replace(' \n\n\n\n', '')
            
            updated_steps = [f'{step} \n\n\n\n' for step in steps]
            steps_all = f'{question} \n\n' + ''.join(updated_steps)
            
            flattened.append({
                'steps_all': steps_all,
                'q_idx': q_idx,
                'cot_idx': cot_idx
            })
    
    return flattened

def reconstruct_results(dataset, reward_results):
    """Reconstruct original dataset format with rewards"""
    # Initialize results structure
    results = []
    for data in dataset:
        result_data = data.copy()
        result_data['rewards'] = [None] * len(data['cots'])  # Pre-allocate
        results.append(result_data)
    
    # Fill in rewards at correct positions
    for reward_item in reward_results:
        q_idx = reward_item['q_idx']
        cot_idx = reward_item['cot_idx']
        reward = reward_item['reward']
        results[q_idx]['rewards'][cot_idx] = reward
    
    return results

def process_gpu_batch(gpu_id, dataset, args, temp_file=None):
    """Process a batch of data on specific GPU"""
    print(f"Process {gpu_id}: Initializing PRM on GPU {gpu_id}...")
    
    # Initialize PRM model with specific GPU device
    prm = PRM(model_id=args.model_id, aggregation="full", device=torch.device(f'cuda:{gpu_id}'))
    
    # Step 1: Flatten all CoTs from this process's dataset
    flattened_data = flatten_all_data(dataset)
    print(f"Process {gpu_id}: PRM initialized. Starting processing...")
    print(f"Process {gpu_id}: Total CoTs to process: {len(flattened_data)}")
    
    # Step 2: Process in real batches of batch_size
    reward_results = []
    for i in tqdm(
        range(0, len(flattened_data), args.per_device_batch_size), 
        desc=f'Process {gpu_id}: Processing CoT batches'
    ):
        batch = flattened_data[i:i + args.per_device_batch_size]
        batch_steps = [item['steps_all'] for item in batch]
        
        # Get rewards for this batch
        batch_rewards = prm(batch_steps)
        
        # Store rewards with their original indices
        for reward, item in zip(batch_rewards, batch):
            reward_results.append({
                'reward': reward,
                'q_idx': item['q_idx'],
                'cot_idx': item['cot_idx']
            })
    
    # Step 3: Reconstruct original format
    print(f"Process {gpu_id}: Reconstructing results...")
    output_results = reconstruct_results(dataset, reward_results)
    
    if temp_file is not None:
        with open(temp_file, "w") as f:
            json.dump(output_results, f, indent=4)
        print(f"Process {gpu_id}: Results saved to {temp_file}")
    else:
        return output_results

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Process rewards for PRM models')
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default="./local_datasets/test")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument("--category", type=str, default="law", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', "all", 'gsm8k', 'math'
    }, help="Category of problems to process")
    parser.add_argument('--per_device_batch_size', type=int, default=8, help="Batch size for CoT processing")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU configuration
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs, creating {num_gpus} processes (1 GPU per process)")
    
    # Only set multiprocessing start method if using multiple processes
    if num_gpus > 1:
        mp.set_start_method('spawn', force=True)

    # Set categories
    if args.category == "all":
        category_list = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                        'history', 'economics', 'math', 'business', 'philosophy', 
                        'health', 'engineering', 'computer_science', 'other']
    else:
        category_list = [args.category]

    # Process each category
    for category in category_list:
        print(f"Loading dataset for category: {category}...")
        
        with open(os.path.join(args.input_dir, f"{category}_dataset.json"), "r") as f:            
            dataset = json.load(f)
        
        if args.debug:
            dataset = dataset[:num_gpus * 2]  # Small dataset for debugging
        
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
                    temp_file = os.path.join(args.output_dir, f"{category}_temp_file_{gpu_id}.json")
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
            output_results = process_gpu_batch(0, dataset, args, None)

        # Save final results
        output_file = os.path.join(args.output_dir, 
            f"debug_{category}_dataset.json" if args.debug else f"eval_{category}_dataset.json")
        with open(output_file, "w") as f:
            json.dump(output_results, f, indent=4)
            
        print(f"Results for {category} saved to {output_file}")
        if num_gpus > 1:
            print(f"Processed {len(output_results)} items using {num_gpus} processes with 1 GPU each")
        else:
            print(f"Processed {len(output_results)} items using single process with 1 GPU")

if __name__ == '__main__':
    main()