from models.llm import ModelInference

def create_checklist(response_dict: dict, model_id: str, topic: str) -> dict: 
    """Create checklist for evaluation from Language Model"""
    
    checklist_dict = {}
    
    # Read Prompt template for checklist generation
    prompt_folder = "./prompts"
    with open(prompt_folder + '/checklist/checklist.txt') as f:
        prompt = f.read()
    
    # Load the LM
    llm = ModelInference(model_id=model_id, quantization='fp8')
        
    for key, response in response_dict.items():
        prompt_type , model_name = key.split('|')[0] , key.split('|')[1]
        
        # Open instruction prompt and format
        with open(f'{prompt_folder}/instructions/{prompt_type}.txt') as f:
            query = f.read()
        query.replace('[TOPIC]', topic)
        
        # Replace checklist prompt's placeholders
        prompt = prompt.replace('{user_query}', query)
        prompt = prompt.replace('{reference_response}', response)

        # Generate Checklist
        response = llm.generate(query=prompt, model_name=model_name, remove_cot=True)

        # Save the response
        with open(f"output/checklist/{key}.txt", encoding='utf-8', mode='w') as f:
            f.write(response)
        
        checklist_dict[key] = response
    
    return checklist_dict