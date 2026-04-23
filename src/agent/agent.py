from models.chat_model import chat_model
from persona.prompt_builder import build_prompt

input_text = "这篇文章提出了什么新算法？"

prompt = build_prompt(input_text)

response = chat_model.stream(prompt)
for chunk in response:
    print(chunk.content, end="", flush=True)

 