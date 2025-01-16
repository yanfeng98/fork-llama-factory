

## Table of Contents

- [LoRA Fine-Tuning](#lora-fine-tuning)
- [QLoRA Fine-Tuning](#qlora-fine-tuning)
- [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
- [Merging LoRA Adapters and Quantization](#merging-lora-adapters-and-quantization)
- [Inferring LoRA Fine-Tuned Models](#inferring-lora-fine-tuned-models)
- [Extras](#extras)

Use `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) to choose computing devices.

## Examples

### QLoRA Fine-Tuning

### Merging LoRA Adapters and Quantization

#### Merge LoRA Adapters

Note: DO NOT use quantized model or `quantization_bit` when merging LoRA adapters.

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### Quantizing Model using AutoGPTQ

```bash
llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```