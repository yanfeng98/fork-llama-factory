#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/qwen_lora_pt.yaml
