"""
$ pip install vllm
$ vllm serve Qwen/Qwen2.5-Coder-0.5B-Instruct --tensor-parallel-size 8
$ curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
"""

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    prompt="write a quick sort algorithm in Python.",
    max_tokens=256
)

print(f"Completion result:\n{completion}")
print(f"{'-'*42}")
print(f"response:\n{completion.choices[0].text}")

print(f"{'-'*42}{'qwq'}{'-'*42}")

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "write a quick sort algorithm in Python."},
    ]
)

print(f"Chat response:\n{chat_response}")
print(f"{'-'*42}")
print(chat_response.choices[0].message.content)

print(f"{'-'*42}{'qwq'}{'-'*42}")

models = client.models.list()
model = models.data[0].id

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }],
    model=model,
)

print(f"Chat completion results:\n{chat_completion}")
print(f"{'-'*42}")
print(chat_completion.choices[0].message.content)