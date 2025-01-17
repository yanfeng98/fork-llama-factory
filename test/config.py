from transformers import AutoConfig

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
config = AutoConfig.from_pretrained(model_name)

# eager
print(config._attn_implementation)
# None
print(config.rope_scaling)
# 32768
print(config.max_position_embeddings)

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")
config = AutoConfig.from_pretrained(model_dir)

# eager
print(config._attn_implementation)
# {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
print(config.rope_scaling)
# 131072
print(config.max_position_embeddings)