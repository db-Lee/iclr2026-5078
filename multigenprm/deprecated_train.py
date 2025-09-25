import argparse
import os
import yaml
from easydict import EasyDict as edict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from multigenprm.trainer.sft_with_kl import SFTTrainerWIthKL
from multigenprm.data.formats import CHAT_TEMPLATE
from multigenprm.utils import load_dataset, preprocess_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_model_and_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        attn_implementation="flash_attention_2"
    )
    
    if 'lora_config' in configs:
        print('Using Lora')
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek R1-Distill with LoRA support")
    
    parser = argparse.ArgumentParser(description='Training script for traing Llama PRM')
    parser.add_argument('-c','--config', type=str, help='Path to config json', default='./configs/llama_prm800k.yml')
    parser.add_argument('--amlt_output_dir', type=str, help='AMLT_OUTPUT_DIR', default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            
    # set wandb project in which to store logs
    if 'wandb_project' in configs:
        os.environ['WANDB_PROJECT'] = configs.wandb_project

    # AMLT_OUTPUT_DIR
    amlt_output_dir = args.amlt_output_dir
    if amlt_output_dir is not None:
        # Remove leading './' if present for cleaner path joining
        output_dir = configs.training_args["output_dir"]
        if output_dir.startswith("./"):
            output_dir = output_dir[2:]
        
        configs.training_args["output_dir"] = os.path.join(amlt_output_dir, output_dir)
    
    ### effective batch size ###
    num_gpus = torch.cuda.device_count()
    configs.training_args["per_device_train_batch_size"] = args.per_device_train_batch_size
    configs.training_args["gradient_accumulation_steps"] = (
        configs.training_args["effective_batch_size"] // \
            (num_gpus*configs.training_args["per_device_train_batch_size"]))
    del configs.training_args["effective_batch_size"]
    print(configs)
        
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # dataset
    train_dataset = load_dataset(args.data_path, args.category)
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    
    # Training configuration
    training_args = SFTConfig(**configs.training_args)
    
    # Initialize trainer
    trainer_class = SFTTrainerWIthKL if configs.kl_coeff else SFTTrainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
        "processing_class": tokenizer
    }
    if args.kl_coeff > 0:
        trainer_kwargs["kl_coeff"] = args.kl_coeff
    
    trainer = trainer_class(**trainer_kwargs)
    
    if trainer.is_world_process_zero():
        print("Sample training text:")
        print(tokenizer.decode(trainer.train_dataset[0:100]["input_ids"][5]))
        print("\n" * 4)
    
    # train
    trainer.train()
    
    # save
    if trainer.is_world_process_zero():
        model.save_pretrained(configs.training_args["output_dir"], safe_serialization=True)
        tokenizer.save_pretrained(configs.training_args["output_dir"])

if __name__ == "__main__":
    main()