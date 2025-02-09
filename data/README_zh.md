[dataset_info.json](dataset_info.json) 包含了所有可用的数据集。如果您希望使用自定义数据集，请**务必**在 `dataset_info.json` 文件中添加*数据集描述*，并通过修改 `dataset: 数据集名称` 配置来使用数据集。

目前我们支持 **alpaca** 格式和 **sharegpt** 格式的数据集。

```json
"数据集名称": {
  "hf_hub_url": "Hugging Face 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "file_name": "该目录下数据集文件夹或文件的名称（若上述参数未指定，则此项必需）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "split": "所使用的数据集切分（可选，默认：train）",
  "folder": "Hugging Face 仓库的文件夹名称（可选，默认：None）",
  "num_samples": "该数据集所使用的样本数量。（可选，默认：None）",
  "columns（可选）": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction）",
  }
}
```

### 预训练数据集

- [样例数据集](c4_demo.json)

在预训练时，只有 `text` 列中的内容会用于模型学习。

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```
