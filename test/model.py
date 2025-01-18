from typing import List
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM

def print_parameters(
    model: "PreTrainedModel",
) -> None:

    for name, param in model.named_parameters():
        print(name)

def find_all_linear_modules(model: "PreTrainedModel") -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    forbidden_modules = {"lm_head"}

    module_names = set()
    for name, module in model.named_modules():
        print(name)
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    print("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
# None
print(getattr(model, "quantization_method", None))
# True
print(isinstance(model, PreTrainedModel))
print(print_parameters(model))
print(f"\n{'*'*64}\n")
# Found linear modules: q_proj,gate_proj,v_proj,k_proj,up_proj,down_proj,o_proj
find_all_linear_modules(model)

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
print(print_parameters(model))
print(f"\n{'*'*64}\n")
# Found linear modules: q_proj,gate_proj,v_proj,k_proj,up_proj,down_proj,o_proj
find_all_linear_modules(model)