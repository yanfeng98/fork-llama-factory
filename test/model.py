from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# None
print(getattr(model, "quantization_method", None))
# True
print(isinstance(model, PreTrainedModel))

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
# QuantizationMethod.BITS_AND_BYTES
print(getattr(model, "quantization_method", None))
# True
print(isinstance(model, PreTrainedModel))

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)
# None
print(getattr(model, "quantization_method", None))
# True
print(isinstance(model, PreTrainedModel))

model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_8bit=True)
# QuantizationMethod.BITS_AND_BYTES
print(getattr(model, "quantization_method", None))
# True
print(isinstance(model, PreTrainedModel))