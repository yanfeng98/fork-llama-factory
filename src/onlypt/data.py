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
import json
from itertools import chain
from functools import partial
from dataclasses import dataclass
from typing import Callable, Tuple, Any, Dict, List, Literal, Optional, Sequence, Union, TypedDict

import numpy as np
from datasets import Dataset, IterableDataset
from datasets import DatasetDict, load_dataset, concatenate_datasets, interleave_datasets
from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments

from .extras import logging
from .extras.constants import FILEEXT2TYPE, DATA_CONFIG
from .hparams import DataArguments, ModelArguments

logger = logging.get_logger(__name__)

# Set a numpy random state for FIM transformations
np_rng = np.random.RandomState(seed=42)

@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "file"]
    dataset_name: str
    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset"]]


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, tokenizer, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, tokenizer, is_eval=True
        )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module

def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)

def get_dataset_list(dataset_names: Optional[Sequence[str]], dataset_dir: str) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    """

    config_path = os.path.join(dataset_dir, DATA_CONFIG)

    with open(config_path) as f:
        dataset_info = json.load(f)

    dataset_list: List["DatasetAttr"] = []
    for name in dataset_names:

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]

        if has_hf_url:
            dataset_attr = DatasetAttr(load_from="hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        else:
            dataset_attr = DatasetAttr(load_from="file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("split", dataset_info[name], default="train")
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("num_samples", dataset_info[name])
        dataset_attr.set_attr("prompt", dataset_info[name]["columns"])

        dataset_list.append(dataset_attr)

    return dataset_list

def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder
    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        streaming=data_args.streaming,
        num_proc=data_args.preprocessing_num_workers,
        trust_remote_code=True,
    )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)

def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:

    convert_func = partial(convert_data, dataset_attr=dataset_attr)

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )

def convert_data(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
) -> Dict[str, Any]:
    r"""
    Converts alpaca format dataset to the standard format.
    """

    return {"content": example[dataset_attr.prompt]}

def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_once("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")

def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, tokenizer
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        print("eval example:" if is_eval else "training example:")
        print_function(next(iter(dataset)))

    return dataset

def get_preprocess_and_print_func(
    data_args: "DataArguments",
    tokenizer: "PreTrainedTokenizer",
) -> Tuple[Callable, Callable]:
    preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
    print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function

def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    tokenized_examples = tokenizer(examples["content"], add_special_tokens=False)

    if data_args.fim_rate > 0:
        # Get the FIM-specific token ids
        prefix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_prefix_token)
        middle_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_middle_token)
        suffix_tok_id = tokenizer.convert_tokens_to_ids(data_args.fim_suffix_token)

        # The two functions below perform the FIM transformation on the data (either PSM or SPM or PSM+SPM)
        # Don't call fim_transform directly in .map()
        # Adapted from https://github.com/loubnabnl/santacoder-finetuning/blob/main/fim.py#L22C13-L83
        def fim_transform(example):
            """
            This function performs FIM transformation on a single example (list of tokens)
            """
            if np_rng.binomial(1, data_args.fim_rate):
                boundaries = sorted(np_rng.randint(low=0, high=len(example) + 1, size=2))
    
                prefix = example[: boundaries[0]]
                middle = example[boundaries[0] : boundaries[1]]
                suffix = example[boundaries[1] :]
    
                if np_rng.binomial(1, data_args.fim_spm_rate):
                    # Apply Suffix-Prefix-Middle (SPM) transformation
                    transformed_example = [prefix_tok_id, suffix_tok_id] + suffix + [middle_tok_id] + prefix + middle
                else:
                    # Apply Prefix-Suffix-Middle (PSM) transformation
                    transformed_example = [prefix_tok_id] + prefix + [suffix_tok_id] + suffix + [middle_tok_id] + middle
            else:
                transformed_example = example
    
            return transformed_example

        # Below function is the one you are supposed to call in the .map() function
        def apply_fim(examples):
            """
            Apply FIM transformation to a batch of examples
            """
            fim_transform_ids = [fim_transform(ids) for ids in examples["input_ids"]]
            examples["input_ids"] = fim_transform_ids
            # If your application requires custom attention mask, please adjust this function's below line.
            # Since FIM transformation increases the number of tokens in input_ids and labels
            # but leaves the number of tokens unchanged in attention_masks which would cause problems
            examples["attention_mask"] = [[1] * len(mask) for mask in examples["input_ids"]]
            return examples
        
        tokenized_examples = apply_fim(examples=tokenized_examples)

    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    eos_tok_id = tokenizer.convert_tokens_to_ids(eos_token)
    
    def add_eos(examples):
        """
        add eos token
        """
        transform_ids = [ids + [eos_tok_id] for ids in examples["input_ids"]]
        examples["input_ids"] = transform_ids
        examples["attention_mask"] = [[1] * len(mask) for mask in examples["input_ids"]]
        return examples
    
    tokenized_examples = add_eos(examples=tokenized_examples)

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
    
def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})