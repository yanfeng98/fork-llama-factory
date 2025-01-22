from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# True
print(getattr(model, "supports_gradient_checkpointing", False))

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)
# True
print(getattr(model, "supports_gradient_checkpointing", False))