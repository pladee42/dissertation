from models import chat
import os

folder_path = './prompts'
prompt_list = []

# Get the list of files in the folder, sorted by filename
files = sorted(os.listdir(folder_path))

# Loop through each file in the prompts folder
for filename in files:
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"Opening file: {filename}")
            content = file.read()
            prompt_list.append(content)

models = {
    # 'deepseek-r1-1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    # 'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    # 'deepseek-r1-14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    # 'deepseek-r1-32b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-r1-70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    # 'gemma-2-9b': 'google/gemma-2-9b-it',
    # 'gemma-2-27b': 'google/gemma-2-27b-it',
    # 'llama-2-7b': 'unsloth/llama-2-7b-chat',
    # 'llama-2-13b': 'daryl149/llama-2-13b-chat-hf',
    'llama-3-70b': 'unsloth/Llama-3.3-70B-Instruct'
}

topic = 'Polar Bears Rescue by University of Sheffield'

for number, query in enumerate(prompt_list):
    for model_name, model_id in models.items():
        query = query.replace('[TOPIC]', topic)
        response = chat(model_id, query)
        file_name = f'{number+1}|{model_name}.txt'

        with open(f"output/{file_name}", encoding='utf-8', mode='w') as f:
            f.write(response)
