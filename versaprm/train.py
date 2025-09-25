from transformers import  TrainingArguments, Trainer
import yaml
import argparse
from easydict import EasyDict as edict
import os
import torch
from versaprm.utils import *

# set to suppress the following warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# probably fine since we tokenize all data first before passing to trainer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# hacky way to prevent loss/metrics being printed to stdout
# while still enabling logging to wandb
# https://github.com/huggingface/transformers/issues/18093
from transformers.trainer_callback import ProgressCallback
def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
ProgressCallback.on_log = on_log

def main(configs):

    # set wandb project in which to store logs
    if 'wandb_project' in configs:
        os.environ['WANDB_PROJECT'] = configs.wandb_project
    
    configs.category = args.category
    
    if configs.training_args["output_dir"] is None:
        configs.training_args["output_dir"] = args.output_dir
        
    configs.noise_data_ratio = args.noise_data_ratio
    configs.noise_process_ratio = args.noise_process_ratio
        
    if args.gradient_checkpointing:
        configs.training_args["gradient_checkpointing"] = True
        configs.training_args["ddp_find_unused_parameters"] = False

    ### Prepare Model and Tokenizer ###
    print('Preparing Model and Tokenizer')
    model = get_model(configs)
    tokenizer = get_tokenizer(configs.model_id)
    
    ### effective batch size ###
    num_gpus = torch.cuda.device_count()
    configs.training_args["per_device_train_batch_size"] = args.per_device_train_batch_size
    configs.training_args["gradient_accumulation_steps"] = (
        configs.training_args["effective_batch_size"] // \
            (num_gpus*configs.training_args["per_device_train_batch_size"]))
    del configs.training_args["effective_batch_size"]
    print(configs)

    ### Prepare data ###
    print('Preparing and tokenizing data')
    t_dataset, e_dataset = get_datasets(configs, tokenizer)
    collate_fn = get_collate_func(tokenizer)

    ### Get custom loss objective and metrics ###
    prm_compute_loss_func = get_compute_loss_func(tokenizer)
    prm_compute_metrics = get_compute_metrics()
    
    ### training loop ###
    training_args = TrainingArguments(**configs.training_args)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=t_dataset,
        eval_dataset=e_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
        compute_loss_func=prm_compute_loss_func,
        compute_metrics=prm_compute_metrics
    )

    # train
    checkpoint = None
    if 'resume_from_checkpoint' in configs:
        checkpoint = configs.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # save
    if trainer.is_world_process_zero():
        model.save_pretrained(configs.training_args["output_dir"], safe_serialization=True)
        tokenizer.save_pretrained(configs.training_args["output_dir"])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training script for traing Llama PRM')
    parser.add_argument('-c','--config', type=str, help='Path to config json', default='./configs/llama_prm800k.yml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--noise_data_ratio', type=float, default=0.)
    parser.add_argument('--noise_process_ratio', type=float, default=0)
    parser.add_argument("--category", type=str, default="all", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other', "all", 'prm800k'
    })
    parser.add_argument('--gradient_checkpointing', action="store_true")
    args = parser.parse_args()

    with open(args.config) as stream:
        try:
            configs = edict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
                
    main(configs)

