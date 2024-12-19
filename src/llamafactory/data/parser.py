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

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope, use_openmind


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "file"]
    dataset_name: str
    formatting: Literal["alpaca"] = "alpaca"
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


def get_dataset_list(dataset_names: Optional[Sequence[str]], dataset_dir: str) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    """

    config_path = os.path.join(dataset_dir, DATA_CONFIG)

    try:
        with open(config_path) as f:
            dataset_info = json.load(f)
    except Exception as err:
        raise ValueError(f"Cannot open {config_path} due to {str(err)}.")

    dataset_list: List["DatasetAttr"] = []
    for name in dataset_names:

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")
        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("split", dataset_info[name], default="train")
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("num_samples", dataset_info[name])
        dataset_attr.set_attr("prompt", dataset_info[name]["columns"])

        dataset_list.append(dataset_attr)

    return dataset_list
