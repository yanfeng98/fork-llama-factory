from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    print(getattr(config, "auto_map", {}))
    print(tokenizer.init_kwargs.get("auto_map", {}))
    if "AutoConfig" in getattr(config, "auto_map", {}):
        print("config registered")
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        print("model registered")
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        print("tokenizer registered")
        tokenizer.__class__.register_for_auto_class()

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# eager
print(config._attn_implementation)
# None
print(config.rope_scaling)
# 32768
print(config.max_position_embeddings)
# {}
# {}
register_autoclass(config=config, model=model, tokenizer=tokenizer)




from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
config = AutoConfig.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# eager
print(config._attn_implementation)
# {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
print(config.rope_scaling)
# 131072
print(config.max_position_embeddings)
# {}
# {}
register_autoclass(config=config, model=model, tokenizer=tokenizer)