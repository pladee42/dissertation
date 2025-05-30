from models.llm import ModelInference
from utils.output_handler import open_response_files, extract_json_from_markdown
import json

def create_checklist(response_file: str = None,
                     model_id: str = 'deepseek-r1-1.5b', 
                     topic: str = None) -> dict: 
    """Create checklist for evaluation from Language Model"""
    
    checklist_dict = {}
    
    # Read Prompt template for checklist generation
    prompt_folder = "./prompts"
    with open(prompt_folder + '/checklist/checklist.txt') as f:
        prompt = f.read()
        
    response_dict = open_response_files(response_file=response_file)
    
    # Load the LM
    llm = ModelInference(model_id=model_id, quantization='fp8')
        
    for key, response in response_dict.items():
        print(key)
        prompt_type , model_name = key.split('|')[0] , key.split('|')[1]
        
        # Open instruction prompt and format
        with open(f'{prompt_folder}/instructions/{prompt_type}.txt') as f:
            query = f.read()
        query = query.replace('[TOPIC]', topic)
        
        # Replace checklist prompt's placeholders
        prompt = prompt.replace('{user_query}', query)
        prompt = prompt.replace('{reference_response}', response)

        # Generate Checklist
        response = llm.generate(query=prompt, model_name=model_name, remove_cot=True)
        
        # Save the response
        with open(f"output/checklist/{key}.txt", encoding='utf-8', mode='w') as f:
            f.write(response)    

        formatted_response = extract_json_from_markdown(response)
        with open(f"output/checklist/{key}.json", encoding='utf-8', mode='w') as f:
            json.dump(formatted_response, f)
        
        checklist_dict[key] = formatted_response

    return checklist_dict