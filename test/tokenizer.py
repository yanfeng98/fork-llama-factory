from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# eos_token: <|im_end|>
print(f"eos_token: {tokenizer.eos_token}")

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# eos_token: <|eot_id|>
print(f"eos_token: {tokenizer.eos_token}")