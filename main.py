from models.llm import ModelInference
import os
from argparse import ArgumentParser

models_dict = {
    'deepseek-r1-1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'deepseek-r1-14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'deepseek-r1-32b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-r1-70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'gemma-3-12b': 'google/gemma-3-12b-it',
    'gemma-3-27b': 'google/gemma-3-27b-it',
    'llama-2-7b': 'unsloth/llama-2-7b-chat',
    'llama-2-13b': 'daryl149/llama-2-13b-chat-hf',
    'llama-3-70b': 'unsloth/Llama-3.3-70B-Instruct'
}

folder_path = './prompts/instructions'

def open_prompt_files(file_names: str = 'all') -> dict:
    """Open prompts files to experiment"""
    
    prompt_dict = {}
    
    # Check prompt_mode
    if file_names == 'all':
        # Get the list of files in the folder, sorted by filename
        files = sorted(os.listdir(folder_path))
    else:
        files = [file_names]

    # Loop through each file in the prompts folder
    for filename in files:
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                print(f"Opening file: {filename}")
                content = file.read()
                prompt_dict[filename.removesuffix('.txt')] = (content)
    
    return prompt_dict
    
            
def generate_responses(prompt_dict: dict, topic: str, lm_model: str) -> None:
    """Generate responses from Selected Language Models"""
    
    if lm_model == 'all':
        lm_model = models_dict
    else:
        lm_model = {lm_model: models_dict[lm_model]} if lm_model in models_dict else None
    
    # Generate responses for each prompt and model
    for model_name, model_id in lm_model.items():
        # Load Model
        llm = ModelInference(model_id=model_id, quantization='fp8')
        for prompt_type, prompt_content in prompt_dict.items():
            prompt_content = prompt_content.replace('[TOPIC]', topic)
            response = llm.generate(query=prompt_content, model_name=model_name, remove_cot=True)
            file_name = f'{prompt_type}|{model_name}.txt'

            # Save the response
            with open(f"output/responses/{file_name}", encoding='utf-8', mode='w') as f:
                f.write(response)

if __name__ == "__main__":
    # Argument Parsing from Command Line
    parser = ArgumentParser()
    parser.add_argument("--prompt_mode", 
                        type=str, 
                        default='all', 
                        required=False,
                        help="Select prompt type to use. e.g. 01.txt / all")
    parser.add_argument("--language_model", 
                        type=str, 
                        default='deepseek-r1-1.5b', 
                        required=False,
                        choices=models_dict.keys(),
                        help="Select languege model to use. e.g. deepseek-r1-1.5b / all")
    parser.add_argument("--topic", 
                        type=str, 
                        default='Polar Bears Rescue by University of Sheffield', 
                        required=False,
                        help="Topic of the email.")
    args = parser.parse_args()
    
    prompt_dict = open_prompt_files(args.prompt_mode)
    generate_responses(prompt_dict=prompt_dict,
                       topic=args.topic,
                       lm_model=args.language_model)
    
    