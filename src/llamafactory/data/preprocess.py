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

from itertools import chain
from functools import partial
from typing import TYPE_CHECKING, Callable, Tuple, Any, Dict, List

from ..extras import logging

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments

logger = logging.get_logger(__name__)

def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

    if not data_args.packing:

        result = tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)
    else:
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

    return result

def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

def get_preprocess_and_print_func(
    data_args: "DataArguments",
    tokenizer: "PreTrainedTokenizer",
) -> Tuple[Callable, Callable]:
    preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
    print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
