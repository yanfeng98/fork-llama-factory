from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(
    inputs=input_ids,
    streamer=streamer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

thread.start()
generated_text = ""
for new_text in streamer:
    generated_text += new_text
    print(new_text, end="")
print(generated_text)
