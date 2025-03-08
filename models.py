from dotenv import load_dotenv
import os
import torch
import time
import gc
from vllm import LLM, SamplingParams
import re

# Load environment variables once at module level
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

def chat(model_name: str, query: str) -> str:
    """
    Download Open-source models from Hugging Face and Send the query to the model.
    Optimized for maximum performance using vLLM's efficient serving capabilities.
    Supports specialized prompt templates for DeepSeek and Gemma models.
    """
    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Print GPU information
    print(f"\n[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"[INFO] GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Print model info
    print(f"\n[INFO] Using model: {model_name}")
    print(f"[INFO] Loading model {model_name}...")
    
    # Time Tracking
    start_time = time.time()
    
    # Initialize vLLM engine with optimized settings
    llm = LLM(
        model=model_name,
        dtype="bfloat16",  # Use bfloat16 for better performance
        download_dir='./models',
        tensor_parallel_size=torch.cuda.device_count(),  # Use all available GPUs
        trust_remote_code=True
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
        frequency_penalty=0.1  # Similar effect to repetition_penalty
    )
    
    # Detect model family to apply the correct prompt template
    model_lower = model_name.lower()
    
    # Format the prompt based on model type
    if "deepseek" in model_lower:
        # DeepSeek-R1 models don't use system prompts, per documentation
        # All instructions should be in the user prompt (search result #8)
        if "r1" in model_lower:
            # For DeepSeek-R1 models (recommended settings from DeepSeek-R1 docs)
            # Setting temperature within 0.5-0.7 range as recommended
            sampling_params.temperature = 0.6
            
            # If it seems like a math or reasoning problem, add the boxed suggestion
            if any(word in query.lower() for word in ["calculate", "solve", "math", "equation", "proof"]):
                prompt = f"User: {query}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            else:
                prompt = f"User: {query}"
        else:
            # For other DeepSeek models (V2, V3, etc.)
            prompt = f"User: {query}\nAssistant:"
    
    # Google Gemma prompt template
    elif "gemma" in model_lower:
        # Gemma uses a specific format with <start_of_turn> and <end_of_turn> tokens
        # Based on Gemma documentation referenced in search result #7
        prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    
    # LLaMA models
    elif "llama" in model_lower:
        # LLaMA chat format
        prompt = f"<s>[INST] {query} [/INST]"
    
    # Mistral models
    elif "mistral" in model_lower:
        # Mistral chat format
        prompt = f"<s>[INST] {query} [/INST]"
    
    # For ChatML format (e.g., some GPT models)
    elif "gpt" in model_lower:
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    # Default format for other models
    else:
        # Simplified chat format
        prompt = f"User: {query}\nAssistant:"
    
    # Generate response using vLLM's engine
    outputs = llm.generate(prompt, sampling_params)
    
    # Extract the generated text from the output
    output = outputs[0]  # Get first output for a single prompt
    
    # Handle different potential output structures across vLLM versions
    try:
        output_text = output.outputs[0].text
    except (AttributeError, IndexError):
        try:
            output_text = output.outputs.text
        except AttributeError:
            output_text = str(output)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Clean up output if needed
    if '</think>' in output_text:
        output_text = output_text.split('</think>')[1]
    
    # Calculate approximate tokens
    # For a more accurate count, we'd need to use a tokenizer
    input_chars = len(prompt)
    output_chars = len(output_text)
    avg_chars_per_token = 4  # Rough estimation
    input_tokens = int(input_chars / avg_chars_per_token)
    output_tokens = int(output_chars / avg_chars_per_token)
    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0
        
    print(f"\nResponse:\n{output_text}\n")
    print(f"[INFO] {model_name} took: {elapsed_time:.2f} seconds for response")
    print(f"[INFO] Generated approximately {output_tokens} tokens at {tokens_per_second:.2f} tokens/sec\n")
    
    # Clean up to free memory
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return output_text
