from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
gen_config = model.generation_config
# GenerationConfig {
#   "bos_token_id": 151643,
#   "do_sample": true,
#   "eos_token_id": [
#     151645,
#     151643
#   ],
#   "pad_token_id": 151643,
#   "repetition_penalty": 1.05,
#   "temperature": 0.7,
#   "top_k": 20,
#   "top_p": 0.8
# }
print(gen_config)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
gen_config = model.generation_config
# GenerationConfig {
#   "bos_token_id": 151643,
#   "do_sample": true,
#   "eos_token_id": [
#     151645,
#     151643
#   ],
#   "pad_token_id": 151643,
#   "repetition_penalty": 1.05,
#   "temperature": 0.7,
#   "top_k": 20,
#   "top_p": 0.8
# }
print(gen_config)

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)
gen_config = model.generation_config
# GenerationConfig {
#   "bos_token_id": 128000,
#   "do_sample": true,
#   "eos_token_id": [
#     128001,
#     128008,
#     128009
#   ],
#   "temperature": 0.6,
#   "top_p": 0.9
# }
print(gen_config)

model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_8bit=True)
gen_config = model.generation_config
# GenerationConfig {
#   "bos_token_id": 128000,
#   "do_sample": true,
#   "eos_token_id": [
#     128001,
#     128008,
#     128009
#   ],
#   "temperature": 0.6,
#   "top_p": 0.9
# }
print(gen_config)