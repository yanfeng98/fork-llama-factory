import inspect
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# True
print("loss_kwargs" in inspect.signature(model.forward).parameters)

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)
# True
print("loss_kwargs" in inspect.signature(model.forward).parameters)