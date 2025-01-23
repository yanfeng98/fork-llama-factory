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

import os
import random
import subprocess
import sys
from enum import Enum, unique

from . import launcher
from .extras import logging
from .extras.env import VERSION, print_env
from .extras.misc import get_device_count
from .tuner import export_model, run_exp


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli help: show this message                         |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "|   llamafactory-cli env -h: show environment info                   |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to LLaMA Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


@unique
class Command(str, Enum):
    HELP = "help"
    VER = "version"
    ENV = "env"
    TRAIN = "train"
    EXPORT = "export"


def main():
    # 1. sys.argv: ['env/bin/llamafactory-cli', 'train', 'examples/train_lora/qwen_lora_pt.yaml']
    # print(f"1. sys.argv: {sys.argv}\n")
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    # 2. sys.argv: ['env/bin/llamafactory-cli', 'examples/train_lora/qwen_lora_pt.yaml']
    # print(f"2. sys.argv: {sys.argv}\n")
    if command == Command.ENV:
        print_env()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        force_torchrun = os.getenv("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        if force_torchrun or get_device_count() > 1:
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                )
                .format(
                    nnodes=os.getenv("NNODES", "1"),
                    node_rank=os.getenv("NODE_RANK", "0"),
                    nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                )
                .split()
            )
            sys.exit(process.returncode)
        else:
            run_exp()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError(f"Unknown command: {command}.")
