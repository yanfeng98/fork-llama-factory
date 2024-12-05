# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the original GaLore's implementation: https://github.com/jiaweizzhao/GaLore
# and the original LoRA+'s implementation: https://github.com/nikhil-ghosh-berkeley/loraplus
# and the original BAdam's implementation: https://github.com/Ledzy/BAdam
# and the HuggingFace's TRL library: https://github.com/huggingface/trl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from typing_extensions import override

from ..extras import logging
from ..hparams import FinetuningArguments, ModelArguments


if TYPE_CHECKING:
    from transformers import PreTrainedModel, Seq2SeqTrainingArguments

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


class DummyOptimizer(torch.optim.Optimizer):
    r"""
    A dummy optimizer used for the GaLore algorithm.
    """

    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[Dict["torch.nn.Parameter", "torch.optim.Optimizer"]] = None
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    @override
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset

    if model_args.use_unsloth:
        kwargs["tags"] = kwargs["tags"] + ["unsloth"]

    if not training_args.do_train:
        pass
    elif training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(license="other", **kwargs)  # prevent from connecting to hub

def _get_decay_parameter_names(model: "PreTrainedModel") -> List[str]:
    r"""
    Returns a list of names of parameters with weight decay. (weights in non-layernorm layers)
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters

def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info_rank0(f"Using LoRA+ optimizer with loraplus lr ratio {finetuning_args.loraplus_lr_ratio:.2f}.")
    return optimizer

def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:

    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)


def create_custom_scheduler(
    training_args: "Seq2SeqTrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if optimizer is not None and isinstance(optimizer, DummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict: Dict["torch.nn.Parameter", "torch.optim.lr_scheduler.LRScheduler"] = {}

        for param in optimizer_dict.keys():
            scheduler_dict[param] = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer_dict[param],
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
            )

        def scheduler_hook(param: "torch.nn.Parameter"):
            scheduler_dict[param].step()

        for param in optimizer_dict.keys():
            param.register_post_accumulate_grad_hook(scheduler_hook)
