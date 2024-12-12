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

import os
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Dict, Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

CHOICES = ["A", "B", "C", "D"]

DATA_CONFIG = "dataset_info.json"

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")

LAYERNORM_NAMES = {"norm", "ln"}

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

METHODS = ["full", "freeze", "lora"]

PEFT_METHODS = {"lora"}

RUNNING_LOG = "running_log.txt"

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINER_LOG = "trainer_log.jsonl"

TRAINING_ARGS = "training_args.yaml"

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Pre-Training": "pt",
}

VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")

VISION_MODELS = set()


class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
    template: Optional[str] = None,
    vision: bool = False,
) -> None:
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path
        if template is not None and any(suffix in name for suffix in ("-Chat", "-Instruct")):
            DEFAULT_TEMPLATE[name] = template
        if vision:
            VISION_MODELS.add(name)


register_model_group(
    models={
        "GLM-4-9B": {
            DownloadSource.DEFAULT: "THUDM/glm-4-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b",
        },
        "GLM-4-9B-Chat": {
            DownloadSource.DEFAULT: "THUDM/glm-4-9b-chat",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat",
            DownloadSource.OPENMIND: "LlamaFactory/glm-4-9b-chat",
        },
        "GLM-4-9B-1M-Chat": {
            DownloadSource.DEFAULT: "THUDM/glm-4-9b-chat-1m",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat-1m",
        },
    },
    template="glm4",
)


register_model_group(
    models={
        "Llama-3-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
        },
        "Llama-3-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B",
        },
        "Llama-3-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
        },
        "Llama-3-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B-Instruct",
        },
        "Llama-3-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
            DownloadSource.OPENMIND: "LlamaFactory/Llama3-Chinese-8B-Instruct",
        },
        "Llama-3-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-70B-Chinese-Chat",
        },
        "Llama-3.1-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B",
        },
        "Llama-3.1-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B",
        },
        "Llama-3.1-405B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B",
        },
        "Llama-3.1-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        },
        "Llama-3.1-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B-Instruct",
        },
        "Llama-3.1-405B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B-Instruct",
        },
        "Llama-3.1-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-8B-Chinese-Chat",
        },
        "Llama-3.1-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-70B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-70B-Chinese-Chat",
        },
        "Llama-3.2-1B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B",
        },
        "Llama-3.2-3B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B",
        },
        "Llama-3.2-1B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct",
        },
        "Llama-3.2-3B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B-Instruct",
        },
    },
    template="llama3",
)


register_model_group(
    models={
        "Qwen2.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B",
        },
        "Qwen2.5-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B",
        },
        "Qwen2.5-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B",
        },
        "Qwen2.5-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B",
        },
        "Qwen2.5-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B",
        },
        "Qwen2.5-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B",
        },
        "Qwen2.5-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B",
        },
        "Qwen2.5-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "Qwen2.5-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct",
        },
        "Qwen2.5-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct",
        },
        "Qwen2.5-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct",
        },
        "Qwen2.5-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct",
        },
        "Qwen2.5-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct",
        },
        "Qwen2.5-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-0.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-1.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-3B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-AWQ",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-14B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-AWQ",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-32B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-AWQ",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-AWQ",
        },
        "Qwen2.5-Coder-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B",
        },
        "Qwen2.5-Coder-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B",
        },
        "Qwen2.5-Coder-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B",
        },
        "Qwen2.5-Coder-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B",
        },
        "Qwen2.5-Coder-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B",
        },
        "Qwen2.5-Coder-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B",
        },
        "Qwen2.5-Coder-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "Qwen2.5-Coder-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Coder-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "Qwen2.5-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Coder-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B-Instruct",
        },
        "Qwen2.5-Coder-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "Qwen2.5-Math-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-1.5B",
        },
        "Qwen2.5-Math-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-7B",
        },
        "Qwen2.5-Math-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-72B",
        },
        "Qwen2.5-Math-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Math-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-72B-Instruct",
        },
        "QwQ-32B-Preview-Instruct": {
            DownloadSource.DEFAULT: "Qwen/QwQ-32B-Preview",
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B-Preview",
        },
    },
    template="qwen",
)
