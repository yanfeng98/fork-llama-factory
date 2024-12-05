# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
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

from typing import TYPE_CHECKING, List, Sequence, Set, Tuple, Union

import torch
import transformers
import transformers.models

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
transformers_logger = transformers.utils.logging.get_logger(__name__)


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Casts projector output to half precision for fine-tuning quantized VLMs.
    """

    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)
        if model_type == "qwen2_vl":
            mm_projector: "torch.nn.Module" = getattr(getattr(model, "visual"), "merger")
        else:
            return

        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)

def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> Set[str]:
    r"""
    Freezes vision tower and language model for VLM full/freeze tuning.
    """
    forbidden_modules = set()

    return forbidden_modules


def patch_target_modules(
    config: "PretrainedConfig", finetuning_args: "FinetuningArguments", target_modules: Sequence[str]
) -> Union[str, List[str]]:
    r"""
    Freezes vision tower for VLM LoRA tuning.
    """
    model_type = getattr(config, "model_type", None)
    if finetuning_args.freeze_vision_tower:
        if model_type == "qwen2_vl":
            return "^(?!.*visual).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules
    else:
        if model_type == "qwen2_vl":
            return "^(?!.*patch_embed).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules
