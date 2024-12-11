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

from typing import TYPE_CHECKING, Dict

from ..extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


TEMPLATES: Dict[str, "Template"] = {}


def _register_template(
    name: str,
) -> None:
    TEMPLATES[name] = name


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""
    Gets chat template and fixes the tokenizer.
    """
    template = TEMPLATES["empty"]  # placeholder

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

    return template


_register_template(
    name="empty",
)
