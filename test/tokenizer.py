from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# eos_token: <|im_end|>
print(f"eos_token: {tokenizer.eos_token}")

# len(tokenizer): 151665
print(f"len(tokenizer): {len(tokenizer)}")

special_tokens = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]

tokenizer.add_tokens(special_tokens)

# len(tokenizer): 151665
print(f"len(tokenizer): {len(tokenizer)}")

# 151659
print(tokenizer.convert_tokens_to_ids("<|fim_prefix|>"))

# {'input_ids': [[5158, 525, 498, 0], [40, 2776, 6915, 0]], 'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1]]}
print(tokenizer(["how are you!", "I'm fine!"], add_special_tokens=False))

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# eos_token: <|eot_id|>
print(f"eos_token: {tokenizer.eos_token}")

# len(tokenizer): 128256
print(f"len(tokenizer): {len(tokenizer)}")

special_tokens = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]

tokenizer.add_tokens(special_tokens)

# len(tokenizer): 128259
print(f"len(tokenizer): {len(tokenizer)}")