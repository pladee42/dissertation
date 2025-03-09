from models.llm import ModelInference

def generate_scores(response_dict: dict, checklist_dict: dict, model_id: str, topic: str) -> dict: 
    """Generate judgement score for each checklist from Language Model"""
    
    judge_dict = {}
    
    # Read Prompt template for judgement score generation
    prompt_folder = "./prompts"
    with open(prompt_folder + '/judge/judge.txt') as f:
        prompt = f.read()
    
    # Load the LM
    llm = ModelInference(model_id=model_id, quantization='fp8')
        
    for key, response in response_dict.items():
        prompt_type , model_name = key.split('|')[0] , key.split('|')[1]
        # Open instruction prompt and format
        with open(f'{prompt_folder}/instructions/{prompt_type}.txt') as f:
            query = f.read()
        query.replace('[TOPIC]', topic)
        
        for _ , checklist in checklist_dict.items():
            
        
            # Replace checklist prompt's placeholders
            prompt = prompt.replace('{user_query}', query)
            prompt = prompt.replace('{model_output}', response)
            prompt = prompt.replace('{checklist}', checklist)

            # Generate Checklist
            response = llm.generate(query=prompt, model_name=model_name, remove_cot=True)

            # Save the response
            with open(f"output/judge/{key}.txt", encoding='utf-8', mode='w') as f:
                f.write(response)

    judge_dict[key] = response
    
    return judge_dict