## 目录

- [LoRA 微调](#lora-微调)
- [QLoRA 微调](#qlora-微调)
- [全参数微调](#全参数微调)
- [合并 LoRA 适配器与模型量化](#合并-lora-适配器与模型量化)
- [推理 LoRA 模型](#推理-lora-模型)
- [杂项](#杂项)

使用 `CUDA_VISIBLE_DEVICES`（GPU）选择计算设备。

## 示例

### LoRA 微调

#### （增量）预训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen_lora_pt.yaml
```

#### 多机指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### 使用 DeepSpeed ZeRO-3 平均分配显存

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen_lora_pt_ds0.yaml
```

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen_lora_pt_ds3.yaml
```

### QLoRA 微调

#### 基于 4/8 比特 Bitsandbytes/HQQ/EETQ 量化进行指令监督微调（推荐）

```bash
llamafactory-cli train examples/train_qlora/qwen_lora_pt_otfq.yaml
```

#### 基于 4/8 比特 GPTQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml
```

#### 基于 4 比特 AWQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
```

#### 基于 2 比特 AQLM 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### 全参数微调

#### 在单机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen_full_pt_ds3.yaml
```

#### 在多机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

### 合并 LoRA 适配器与模型量化

#### 合并 LoRA 适配器

注：请勿使用量化后的模型或 `quantization_bit` 参数来合并 LoRA 适配器。

```bash
llamafactory-cli export examples/merge_lora/qwen_lora_pt.yaml
```

### 杂项

#### FSDP+QLoRA 微调

```bash
accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/qwen_lora_pt.yaml
```
