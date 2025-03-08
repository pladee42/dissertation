# from dissertation.models import chat
import os
from pathlib import Path

data_folder = Path("dissertation/prompts/")
file_to_open = data_folder / "01.txt"

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('..', cur_path)
print(file_to_open)
print(new_path)
with open(file_to_open) as f:
    prompt = f.read()
    
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