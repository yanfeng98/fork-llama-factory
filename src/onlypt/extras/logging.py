# Copyright 2024 Optuna, HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py
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

import logging
import os
import sys
import threading
from functools import lru_cache
from typing import Optional


_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


class _Logger(logging.Logger):
    r"""
    A logger that supports info_rank0 and warning_once.
    """

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)


def get_logger(name: Optional[str] = None) -> "_Logger":
    r"""
    Returns a logger with the specified name. It it not supposed to be accessed externally.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)

def _configure_library_root_logger() -> None:
    r"""
    Configures root logger using a stdout stream handler with an explicit format.
    """
    global _default_handler

    with _thread_lock:
        if _default_handler:  # already configured
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = logging.getLogger(__name__.split(".")[0])
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_log_level)
        library_root_logger.propagate = False


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_once = warning_once
