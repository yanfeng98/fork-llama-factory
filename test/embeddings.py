from transformers import AutoModelForCausalLM

def fun(model):

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    module_names = set()
    for name, module in model.named_modules():
        if module in [input_embeddings, output_embeddings]:
            module_names.add(name.split(".")[-1])
    print(module_names)

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# {'lm_head', 'embed_tokens'}
fun(model=model)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

# {'lm_head', 'embed_tokens'}
fun(model=model)

from modelscope import snapshot_download

model_dir: str = snapshot_download("LLM-Research/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(model_dir)

# {'lm_head', 'embed_tokens'}
fun(model=model)

model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_8bit=True)

# {'lm_head', 'embed_tokens'}
fun(model=model)