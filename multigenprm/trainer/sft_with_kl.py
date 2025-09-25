from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from peft import PeftConfig, PeftModel
from transformers import (AutoModelForCausalLM, BaseImageProcessor,
                          DataCollator, FeatureExtractionMixin,
                          PreTrainedModel, PreTrainedTokenizerBase,
                          ProcessorMixin, TrainingArguments)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl import SFTConfig, SFTTrainer

class SFTTrainerWIthKL(SFTTrainer):
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor,
                  FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer],
                          Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer],
                                                 dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[
            torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[
            dict], str], Callable[[dict], list[str]]]] = None,
        kl_coeff: float = 0.1,  # Coefficient for KL divergence penalty
    ):
        # We need to manually handle the model if it's a PeftModel before calling super().__init__
        # because SFTTrainer will wrap it again if peft_config is provided.
        self.is_lora = isinstance(model, PeftModel)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        self.kl_coeff = kl_coeff
        if kl_coeff > 0:
            if isinstance(model, PeftModel):
                self.is_lora = True
            else:
                unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
                config = unwrapped_model.config
                self.ref_model = AutoModelForCausalLM.from_config(config)

                state_dict = unwrapped_model.state_dict()
                self.ref_model.load_state_dict(state_dict)
                self.ref_model = self.ref_model.to(self.model.device)
                self.ref_model.eval()
        else:
            print("KL coefficient is zero. It fall backs to the original SFT.")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        if self.kl_coeff > 0:
            mode = "train" if self.model.training else "eval"
            logits = outputs.logits

            if self.is_lora:
                unwrapped_model = model.module if hasattr(model, "module") else model
                try:
                    with torch.no_grad():
                        # Deactivate adapters to get the base model's output
                        unwrapped_model.disable_adapter_layers()
                        ref_outputs = unwrapped_model(**inputs, use_cache=False)
                finally:
                    # Always re-enable the adapters for the main training pass
                    unwrapped_model.enable_adapter_layers()
            else:
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs, use_cache=False)

            ref_logits = ref_outputs.logits

            # Create a mask to ignore padded parts of the sequence (-100 in labels)
            padding_mask = (inputs["labels"] != -100)

            # Ensure the mask is on the same device as the logits
            padding_mask = padding_mask.to(logits.device)

            # Reshape logits and mask for filtering
            vocab_size = logits.size(-1)

            # Filter out the logits and reference logits at padded positions
            active_logits = logits.view(-1, vocab_size)[padding_mask.view(-1)]
            active_ref_logits = ref_logits.view(-1, vocab_size)[padding_mask.view(-1)]

            # Calculate KL divergence
            # `kl_div` expects log-probabilities for the input and probabilities for the target.
            log_probs = F.log_softmax(active_logits, dim=-1)
            ref_probs = F.softmax(active_ref_logits, dim=-1)

            # `reduction='batchmean'` computes the mean loss per sample in the batch.
            # Since we have filtered for active tokens, this is the mean KL divergence per token.
            kl_div = F.kl_div(log_probs, ref_probs, reduction="batchmean")

            # 4. COMBINE THE LOSSES
            # Add the scaled KL divergence to the supervised loss
            self._metrics[mode]["kl"].append(kl_div.item())
            loss = base_loss + self.kl_coeff * kl_div
        else:
            loss = base_loss

        return (loss, outputs) if return_outputs else loss
