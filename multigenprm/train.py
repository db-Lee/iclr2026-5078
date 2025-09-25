import argparse
import os
import wandb

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from multigenprm.trainer.sft_with_kl import SFTTrainerWIthKL
from multigenprm.data.formats import CHAT_TEMPLATE
from multigenprm.utils import load_train_dataset, preprocess_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" \
            if not args.no_flash_attn else "sdpa",
        device_map=None
    )
    
    if args.use_lora:
        model.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek R1-Distill with LoRA support")
    
    parser.add_argument("--data_path", type=str, default="./local_datasets")
    parser.add_argument("--output_dir", type=str, default="./training_results")
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--category", type=str, default="law", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', "all", 'prm800k'
    })
    parser.add_argument("--task_type", type=str, default="orm", choices={"orm", "prm"})
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "epoch"])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--kl_coeff", type=float, default=0.0)
    parser.add_argument("--no_flash_attn", action="store_true")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="GenPRM")
    parser.add_argument("--exp_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    
    # Setup directories and batch size calculations
    args.output_dir = os.path.join(args.output_dir, args.exp_name or "debug")
    world_size = accelerator.num_processes
    args.gradient_accumulation_steps = max(1, args.effective_batch_size // (args.per_device_batch_size * world_size))
    actual_effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    
    # Initialize wandb and print info
    if accelerator.is_main_process:
        print(f"Training: {world_size} GPUs, batch {args.per_device_batch_size}, "
              f"grad acc {args.gradient_accumulation_steps}, effective {actual_effective_batch_size}")
        if args.use_lora:
            print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        
        wandb.init(
            project=args.wandb_project, 
            name=args.exp_name,
            config={**vars(args), "world_size": world_size, "actual_effective_batch_size": actual_effective_batch_size}
        )
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)    
    if args.use_lora and accelerator.is_main_process:
        model.print_trainable_parameters()
    
    # dataset
    train_dataset = load_train_dataset(args.task_type, args.data_path, args.category)
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        gradient_checkpointing=not args.use_lora,
        dataloader_pin_memory=True,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_only_model=True,
        report_to="wandb",
        remove_unused_columns=False,
        packing=False,
        ddp_find_unused_parameters=False,
        max_length=None,
        completion_only_loss=True,
        weight_decay=args.weight_decay
    )
    
    # Initialize trainer
    trainer_class = SFTTrainerWIthKL if args.kl_coeff > 0 else SFTTrainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
        "processing_class": tokenizer
    }
    if args.kl_coeff > 0:
        trainer_kwargs["kl_coeff"] = args.kl_coeff
    
    trainer = trainer_class(**trainer_kwargs)
    
    if accelerator.is_main_process:
        print("Sample training text:")
        print(tokenizer.decode(trainer.train_dataset[0:100]["input_ids"][5]))
        print("\n" * 4)
    
    # Train and save
    trainer.train()
    
    if accelerator.is_main_process:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    main()