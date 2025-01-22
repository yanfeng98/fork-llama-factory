# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/import_utils.py
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

import importlib.metadata
import importlib.util


def _is_package_available(name: str) -> bool:
    """
    检查指定包是否可用。
    
    Args:
        name (str): 要检查的包名。
    
    Returns:
        bool: 如果包可用，则返回True；否则返回False。
    """
    return importlib.util.find_spec(name) is not None

def is_matplotlib_available():
    return _is_package_available("matplotlib")