![# LLaMA Factory](assets/logo.png)

\[ [English](README.md) | 中文 \]

**微调大模型可以像这样轻松…**

选择你的打开方式：

- **入门教程**：https://zhuanlan.zhihu.com/p/695287607
- **框架文档**：https://llamafactory.readthedocs.io/zh-cn/latest/

## 模型

| 模型名                                                            | 模型大小                          | Template         |
| ----------------------------------------------------------------- | -------------------------------- | ---------------- |
| [Baichuan 2](https://huggingface.co/baichuan-inc)                 | 7B/13B                           | baichuan2        |
| [BLOOM/BLOOMZ](https://huggingface.co/bigscience)                 | 560M/1.1B/1.7B/3B/7.1B/176B      | -                |
| [ChatGLM3](https://huggingface.co/THUDM)                          | 6B                               | chatglm3         |
| [Command R](https://huggingface.co/CohereForAI)                   | 35B/104B                         | cohere           |
| [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)         | 7B/16B/67B/236B                  | deepseek         |
| [Falcon](https://huggingface.co/tiiuae)                           | 7B/11B/40B/180B                  | falcon           |
| [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)          | 2B/7B/9B/27B                     | gemma            |
| [GLM-4](https://huggingface.co/THUDM)                             | 9B                               | glm4             |
| [Index](https://huggingface.co/IndexTeam)                         | 1.9B                             | index            |
| [InternLM2/InternLM2.5](https://huggingface.co/internlm)          | 7B/20B                           | intern2          |
| [Llama](https://github.com/facebookresearch/llama)                | 7B/13B/33B/65B                   | -                |
| [Llama 2](https://huggingface.co/meta-llama)                      | 7B/13B/70B                       | llama2           |
| [Llama 3-3.2](https://huggingface.co/meta-llama)                  | 1B/3B/8B/70B                     | llama3           |
| [Llama 3.2 Vision](https://huggingface.co/meta-llama)             | 11B/90B                          | mllama           |
| [LLaVA-1.5](https://huggingface.co/llava-hf)                      | 7B/13B                           | llava            |
| [LLaVA-NeXT](https://huggingface.co/llava-hf)                     | 7B/8B/13B/34B/72B/110B           | llava_next       |
| [LLaVA-NeXT-Video](https://huggingface.co/llava-hf)               | 7B/34B                           | llava_next_video |
| [MiniCPM](https://huggingface.co/openbmb)                         | 1B/2B/4B                         | cpm/cpm3         |
| [Mistral/Mixtral](https://huggingface.co/mistralai)               | 7B/8x7B/8x22B                    | mistral          |
| [OLMo](https://huggingface.co/allenai)                            | 1B/7B                            | -                |
| [PaliGemma](https://huggingface.co/google)                        | 3B                               | paligemma        |
| [Phi-1.5/Phi-2](https://huggingface.co/microsoft)                 | 1.3B/2.7B                        | -                |
| [Phi-3](https://huggingface.co/microsoft)                         | 4B/7B/14B                        | phi              |
| [Pixtral](https://huggingface.co/mistralai)                       | 12B                              | pixtral          |
| [Qwen (1-2.5) (Code/Math/MoE)](https://huggingface.co/Qwen)       | 0.5B/1.5B/3B/7B/14B/32B/72B/110B | qwen             |
| [Qwen2-VL](https://huggingface.co/Qwen)                           | 2B/7B/72B                        | qwen2_vl         |
| [Skywork o1](https://huggingface.co/Skywork)                      | 8B                               | skywork_o1       |
| [StarCoder 2](https://huggingface.co/bigcode)                     | 3B/7B/15B                        | -                |
| [XVERSE](https://huggingface.co/xverse)                           | 7B/13B/65B                       | xverse           |
| [Yi/Yi-1.5 (Code)](https://huggingface.co/01-ai)                  | 1.5B/6B/9B/34B                   | yi               |
| [Yi-VL](https://huggingface.co/01-ai)                             | 6B/34B                           | yi_vl            |
| [Yuan 2](https://huggingface.co/IEITYuan)                         | 2B/51B/102B                      | yuan             |

> [!NOTE]
> 对于所有“基座”（Base）模型，`template` 参数可以是 `default`, `alpaca`, `vicuna` 等任意值。但“对话”（Instruct/Chat）模型请务必使用**对应的模板**。
>
> 请务必在训练和推理时采用**完全一致**的模板。

项目所支持模型的完整列表请参阅 [constants.py](src/llamafactory/extras/constants.py)。

您也可以在 [template.py](src/llamafactory/data/template.py) 中添加自己的对话模板。

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

<details><summary>指令微调数据集</summary>

- [Identity (en&zh)](data/identity.json)
- [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)
- [Alpaca GPT4 (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Glaive Function Calling V2 (en&zh)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
- [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [UltraChat (en)](https://github.com/thunlp/UltraChat)
- [OpenPlatypus (en)](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
- [OpenOrca (en)](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [SlimOrca (en)](https://huggingface.co/datasets/Open-Orca/SlimOrca)
- [MathInstruct (en)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [Wiki QA (en)](https://huggingface.co/datasets/wiki_qa)
- [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
- [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [deepctrl (en&zh)](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [Advertise Generating (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
- [ShareGPT Hyperfiltered (en)](https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k)
- [ShareGPT4 (en&zh)](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
- [UltraChat 200k (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [AgentInstruct (en)](https://huggingface.co/datasets/THUDM/AgentInstruct)
- [LMSYS Chat 1M (en)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [Evol Instruct V2 (en)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)
- [Cosmopedia (en)](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- [STEM (zh)](https://huggingface.co/datasets/hfl/stem_zh_instruction)
- [Ruozhiba (zh)](https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo)
- [Neo-sft (zh)](https://huggingface.co/datasets/m-a-p/neo_sft_phase2)
- [Magpie-Pro-300K-Filtered (en)](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered)
- [Magpie-ultra-v0.1 (en)](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)
- [WebInstructSub (en)](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub)
- [OpenO1-SFT (en&zh)](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)
- [LLaVA mixed (en&zh)](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)
- [Pokemon-gpt4o-captions (en&zh)](https://huggingface.co/datasets/jugg1024/pokemon-gpt4o-captions)
- [Open Assistant (de)](https://huggingface.co/datasets/mayflowergmbh/oasst_de)
- [Dolly 15k (de)](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de)
- [Alpaca GPT4 (de)](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de)
- [OpenSchnabeltier (de)](https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de)
- [Evol Instruct (de)](https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de)
- [Dolphin (de)](https://huggingface.co/datasets/mayflowergmbh/dolphin_de)
- [Booksum (de)](https://huggingface.co/datasets/mayflowergmbh/booksum_de)
- [Airoboros (de)](https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de)
- [Ultrachat (de)](https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de)

</details>

<details><summary>偏好数据集</summary>

- [DPO mixed (en&zh)](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)
- [UltraFeedback (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [RLHF-V (en)](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset)
- [VLFeedback (en)](https://huggingface.co/datasets/Zhihui/VLFeedback)
- [Orca DPO Pairs (en)](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [Orca DPO (de)](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)
- [KTO mixed (en)](https://huggingface.co/datasets/argilla/kto-mix-15k)

</details>

部分数据集的使用需要确认，我们推荐使用下述命令登录您的 Hugging Face 账户。

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 如何使用

### 安装 LLaMA Factory

> [!IMPORTANT]
> 此步骤为必需。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

可选的额外依赖项：torch、torch-npu、metrics、deepspeed、liger-kernel、bitsandbytes、hqq、eetq、gptq、awq、aqlm、vllm、galore、badam、adam-mini、qwen、modelscope、openmind、quality

> [!TIP]
> 遇到包冲突时，可使用 `pip install --no-deps -e .` 解决。

### 数据准备

关于数据集文件的格式，请参考 [data/README_zh.md](data/README_zh.md) 的内容。你可以使用 HuggingFace / ModelScope / Modelers 上的数据集或加载本地数据集。

> [!NOTE]
> 使用自定义数据集时，请更新 `data/dataset_info.json` 文件。

### 快速开始

下面三行命令分别对 Llama3-8B-Instruct 模型进行 LoRA **微调**、**推理**和**合并**。

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

高级用法请参考 [examples/README_zh.md](examples/README_zh.md)（包括多 GPU 微调）。

> [!TIP]
> 使用 `llamafactory-cli help` 显示帮助信息。

### LLaMA Board 可视化微调（由 [Gradio](https://github.com/gradio-app/gradio) 驱动）

```bash
llamafactory-cli webui
```

### 利用 vLLM 部署 OpenAI API

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
```

> [!TIP]
> API 文档请查阅[这里](https://platform.openai.com/docs/api-reference/chat/create)。
>
> 示例：[图像理解](scripts/test_image.py) | [工具调用](scripts/test_toolcall.py)

### 从魔搭社区下载

如果您在 Hugging Face 模型和数据集的下载中遇到了问题，可以通过下述方法使用魔搭社区。

```bash
export USE_MODELSCOPE_HUB=1 # Windows 使用 `set USE_MODELSCOPE_HUB=1`
```

将 `model_name_or_path` 设置为模型 ID 来加载对应的模型。在[魔搭社区](https://modelscope.cn/models)查看所有可用的模型，例如 `LLM-Research/Meta-Llama-3-8B-Instruct`。

### 从魔乐社区下载

您也可以通过下述方法，使用魔乐社区下载数据集和模型。

```bash
export USE_OPENMIND_HUB=1 # Windows 使用 `set USE_OPENMIND_HUB=1`
```

将 `model_name_or_path` 设置为模型 ID 来加载对应的模型。在[魔乐社区](https://modelers.cn/models)查看所有可用的模型，例如 `TeleAI/TeleChat-7B-pt`。

### 使用 W&B 面板

若要使用 [Weights & Biases](https://wandb.ai) 记录实验数据，请在 yaml 文件中添加下面的参数。

```yaml
report_to: wandb
run_name: test_run # 可选
```

在启动训练任务时，将 `WANDB_API_KEY` 设置为[密钥](https://wandb.ai/authorize)来登录 W&B 账户。