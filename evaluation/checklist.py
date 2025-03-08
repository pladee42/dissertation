from ..models.models import ModelInference
import os
from pathlib import Path

data_folder = Path("./prompts/")
file_to_open = data_folder / "01.txt"

cur_path = os.path.dirname(__file__)
print(file_to_open)
with open(file_to_open) as f:
    prompt = f.read()

print(prompt)

ModelInference(model_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    
# with open('dissertation/output/1|deepseek-r1-1.5b.txt') as f:
#     response = f.read()

# topic = 'Polar Bears Rescue by University of Sheffield'
    
# prompt = prompt.replace('[TOPIC]', topic)

# model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# query = {
#     'user': prompt,
#     'assistant': response
# }
# response = chat(model_id, query)
# print(query)