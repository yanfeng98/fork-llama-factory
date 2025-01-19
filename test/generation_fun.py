from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# <function GenerationMixin.generate at 0x7f50940d7880>
print(str(model.generate.__func__))

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
# <function GenerationMixin.generate at 0x7f50940d7880>
print(str(model.generate.__func__))

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)
# <function GenerationMixin.generate at 0x7f50940d7880>
print(str(model.generate.__func__))

model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_8bit=True)
# <function GenerationMixin.generate at 0x7f50940d7880>
print(str(model.generate.__func__))