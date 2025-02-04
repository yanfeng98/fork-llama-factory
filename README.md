## 数据集

<details><summary>预训练数据集</summary>

- [Wiki Demo (en)](data/wiki_demo.txt)
- [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- [RedPajama V2 (en)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
- [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [Pile (en)](https://huggingface.co/datasets/EleutherAI/pile)
- [SkyPile (zh)](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- [FineWeb (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb-Edu (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [The Stack (en)](https://huggingface.co/datasets/bigcode/the-stack)
- [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)

</details>

部分数据集的使用需要确认，我们推荐使用下述命令登录您的 Hugging Face 账户。

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 如何使用

### 安装 onlypt

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
$ pip install -e ".[torch,deepspeed,bitsandbytes]" -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

可选的额外依赖项：torch、deepspeed、bitsandbytes、hqq、eetq

> [!TIP]
> 遇到包冲突时，可使用 `pip install --no-deps -e .` 解决。

### 数据准备

关于数据集文件的格式，请参考 [data/README_zh.md](data/README_zh.md) 的内容。你可以使用 HuggingFace 上的数据集或加载本地数据集。

> [!NOTE]
> 使用自定义数据集时，请更新 `data/dataset_info.json` 文件。

### 快速开始

下面命令分别对 Qwen2.5-Coder-0.5B-Instruct 模型进行 LoRA **预训练**和**合并**。

```bash
CUDA_VISIBLE_DEVICES=0 pt train examples/train_lora/qwen_lora_pt.yaml
pt export examples/merge_lora/qwen_lora_pt.yaml
```

高级用法请参考 [examples/README_zh.md](examples/README_zh.md)（包括多 GPU 微调）。

> [!TIP]
> 使用 `pt help` 显示帮助信息。

### 使用 W&B 面板

若要使用 [Weights & Biases](https://wandb.ai) 记录实验数据，请在 yaml 文件中添加下面的参数。

```yaml
report_to: wandb
run_name: test_run # 可选
```

在启动训练任务时，将 `WANDB_API_KEY` 设置为[密钥](https://wandb.ai/authorize)来登录 W&B 账户。