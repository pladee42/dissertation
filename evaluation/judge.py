from models.llm import ModelInference

def generate_scores(response_dict: dict, checklist_dict: dict, model_id: str, topic: str) -> dict: 
    """Generate judgement score for each checklist from Language Model"""
    
    judgment_results = {}
    
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
        query = query.replace('[TOPIC]', topic)
        judgment_results[key] = {}
        
        for _ , checklist in checklist_dict.items():
        
            # Replace checklist prompt's placeholders
            prompt = prompt.replace('{user_query}', query)
            prompt = prompt.replace('{model_output}', response)
            prompt = prompt.replace('{checklist}', checklist)

            # Generate Checklist
            probabilities = llm.compute_yes_no_probability(query=prompt, model_name=model_name)
            
            judgment_results[key][checklist_id] = {
                "question": checklist_dict['question'],
                "yes_probability": probabilities["yes"],
                "no_probability": probabilities["no"],
                "judgment": "Yes" if probabilities["yes"] > probabilities["no"] else "No"
            }

    judge_dict[key] = response
    
    return judge_dict