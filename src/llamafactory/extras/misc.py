# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's PEFT library.
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/peft_model.py
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

import os
from typing import TYPE_CHECKING, Tuple

import torch
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils.versions import require_version

from . import logging


_is_fp16_available = is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False


logger = logging.get_logger(__name__)

def check_dependencies() -> None:
    r"""
    Checks the version of the required packages.
    """
    if os.getenv("DISABLE_VERSION_CHECK", "0").lower() in ["true", "1"]:
        logger.warning_once("Version checking has been disabled, may lead to unexpected behaviors.")
    else:
        require_version("transformers>=4.41.2,<=4.46.1", "To fix: pip install transformers>=4.41.2,<=4.46.1")
        require_version("datasets>=2.16.0,<=3.1.0", "To fix: pip install datasets>=2.16.0,<=3.1.0")
        require_version("accelerate>=0.34.0,<=1.0.1", "To fix: pip install accelerate>=0.34.0,<=1.0.1")
        require_version("peft>=0.11.1,<=0.12.0", "To fix: pip install peft>=0.11.1,<=0.12.0")
        require_version("trl>=0.8.6,<=0.9.6", "To fix: pip install trl>=0.8.6,<=0.9.6")


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> "torch.device":
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_peak_memory() -> Tuple[int, int]:
    r"""
    Gets the peak memory usage for the current device (in Bytes).
    """
    if is_torch_cuda_available():
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    else:
        return 0, 0


def infer_optim_dtype(model_dtype: "torch.dtype") -> "torch.dtype":
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def use_modelscope() -> bool:
    return os.environ.get("USE_MODELSCOPE_HUB", "0").lower() in ["true", "1"]


def use_openmind() -> bool:
    return os.environ.get("USE_OPENMIND_HUB", "0").lower() in ["true", "1"]
