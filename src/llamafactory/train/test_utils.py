# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Set, Tuple, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from ..data import get_dataset, get_template_and_fix_tokenizer
from ..extras.misc import get_current_device
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer


if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraModel
    from transformers import PreTrainedModel


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module", diff_keys: Sequence[str] = []) -> None:
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        if any(key in name for key in diff_keys):
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is False
        else:
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is True


def check_lora_model(model: "LoraModel") -> Tuple[Set[str], Set[str]]:
    linear_modules, extra_modules = set(), set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        elif "modules_to_save" in name:
            extra_modules.add(name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    return linear_modules, extra_modules


def load_train_model(add_valuehead: bool = False, **kwargs) -> "PreTrainedModel":
    model_args, _, _, finetuning_args, _ = get_train_args(kwargs)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    return load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=add_valuehead)

def load_train_dataset(**kwargs) -> "Dataset":
    model_args, data_args, training_args, _, _ = get_train_args(kwargs)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, kwargs["stage"], **tokenizer_module)
    return dataset_module["train_dataset"]
