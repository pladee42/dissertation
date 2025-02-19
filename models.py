from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

load_dotenv(override=True)
hf_token = os.getenv('HUGGINGFACE_TOKEN')

def chat(model_name: str, query: str, hf_token: str = hf_token) -> str:
    """
    Download Open-source models from Hugging Face and Send the query to the model.
    """
    # Time Tracking
    start_time = time.time()  # Start time tracking
    
    # Check if CUDA is available, and move model to GPU if so
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    print(f"\n[INFO] Using device: {device}")

    # Load a pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models', token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                attn_implementation="flash_attention_2",
                                                cache_dir='./models',
                                                token=hf_token).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print(f"\n[INFO] Using model: {model_name}")

    # Test the model by generating text
    template = [
        {"role": "user",
        "content": query}
    ]
    prompt = tokenizer.apply_chat_template(conversation=template,
                                        tokenize=False,
                                        add_generation_prompt=True,
                                        chat_template=query
                                        )

    # Tokenize the input text
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate outputs passed on the tokenized input
    with torch.amp.autocast('cuda'):
        outputs = model.generate(**input_ids, max_new_tokens=4096)

    # Decode and print the output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if '</think>' in output_text:
        output_text = output_text.split('</think>')[1]
        
    print(f"\nResponse:\n{output_text}\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] {model_name} took : {elapsed_time:.2f} seconds for response\n")  # Print elapsed time
    
    return output_text
