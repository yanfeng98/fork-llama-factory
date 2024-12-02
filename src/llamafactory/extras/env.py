# Copyright 2024 luyanfeng
#
# Licensed under the MIT License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform

import accelerate
import datasets
import peft
import torch
import transformers
import trl
from transformers.utils import is_torch_cuda_available


VERSION = "0.9.2.dev0"


def print_env() -> None:
    """
    打印当前环境信息。
    
    Args:
        无参数。
    
    Returns:
        无返回值。
    
    Raises:
        无异常。
    
    """
    info = {
        "`llamafactory` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
        "PEFT version": peft.__version__,
        "TRL version": trl.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info["vLLM version"] = vllm.__version__
    except Exception:
        pass

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")
