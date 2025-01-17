from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# <function PreTrainedTokenizerBase._pad at 0x7f8fe5dae480>
print(str(tokenizer._pad.__func__))
# False
print("PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__))




from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")

print(f"Meta-Llama-3.1-8B-Instruct path: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# <function PreTrainedTokenizerBase._pad at 0x7f8412dae480>
print(str(tokenizer._pad.__func__))
# False
print("PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__))