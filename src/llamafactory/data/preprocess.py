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

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

from .processors.feedback import preprocess_feedback_dataset
from .processors.pretrain import preprocess_pretrain_dataset
from .processors.supervised import (
    print_supervised_dataset_example,
)
from .processors.unsupervised import preprocess_unsupervised_dataset, print_unsupervised_dataset_example


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from .template import Template


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "kto":
        preprocess_func = partial(
            preprocess_feedback_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
