from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "Here is the code:"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,
    return_tensors="pt"
)

text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

# input_ids:
# tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
#             553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
#              13, 151645,    198, 151644,    872,    198,   4934,    264,   3974,
#            3378,  12111,     13, 151645,    198, 151644,  77091,    198,   8420,
#             374,    279,   2038,     25, 151645,    198]])
# text:
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# write a quick sort algorithm.<|im_end|>
# <|im_start|>assistant
# Here is the code:<|im_end|>
print(f"input_ids:\n{input_ids}")
print(f"text:\n{text}")

output_tokens: int = tokenizer("Here is the code:<|im_end|>\n")['input_ids']
prompt_tokens = input_ids[0][:-len(output_tokens)]
decode_prompt: str = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

# output_tokens:
# [8420, 374, 279, 2038, 25, 151645, 198]
# output_tokens length:
# 7
# prompt_tokens:
# tensor([151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
#            553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
#             13, 151645,    198, 151644,    872,    198,   4934,    264,   3974,
#           3378,  12111,     13, 151645,    198, 151644,  77091,    198])
# decode_prompt:
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# write a quick sort algorithm.<|im_end|>
# <|im_start|>assistant
print(f"output_tokens:\n{output_tokens}")
print(f"output_tokens length:\n{len(output_tokens)}")
print(f"prompt_tokens:\n{prompt_tokens}")
print(f"decode_prompt:\n{decode_prompt}")

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

# input_ids:
# tensor([[151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
#             553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
#              13, 151645,    198, 151644,    872,    198,   4934,    264,   3974,
#            3378,  12111,     13, 151645,    198, 151644,  77091,    198]])
# text:
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# write a quick sort algorithm.<|im_end|>
# <|im_start|>assistant
print(f"input_ids:\n{input_ids}")
print(f"text:\n{text}")

# output_tokens:
# [688, 12448, 1843]
# decode_output:
#           ---          
         
# True
#  ---
output_tokens: int = tokenizer(f"{' '*10}---{' '*10}")['input_ids']
decode_output: str = tokenizer.decode(output_tokens, skip_special_tokens=False)
print(f"output_tokens:\n{output_tokens}")
print(f"decode_output:\n{decode_output}")
print(tokenizer.decode([688]))
print(tokenizer.decode([688]) == f"{' '*9}")
print(tokenizer.decode([12448]))

# tokenizer.eos_token: <|im_end|>
# tokenizer.eos_token_id: 151645
# tokenizer.pad_token: <|endoftext|>
# tokenizer.pad_token_id: 151643
# tokenizer.bos_token: None
# tokenizer.bos_token_id: None
print(f"tokenizer.eos_token: {tokenizer.eos_token}")
print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
print(f"tokenizer.bos_token: {tokenizer.bos_token}")
print(f"tokenizer.bos_token_id: {tokenizer.bos_token_id}")

print(tokenizer.chat_template)