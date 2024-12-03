# pip install vllm

from vllm import LLM, SamplingParams

prompts = [
    "write a quick sort algorithm.",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=2048)

llm = LLM(model="Qwen/Qwen2.5-Coder-0.5B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")