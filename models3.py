from dotenv import load_dotenv
from deepspeed.accelerator import get_accelerator
import deepspeed
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import gc
from contextlib import nullcontext

# Load environment variables once at module level
load_dotenv(override=True)
hf_token = os.getenv('HUGGINGFACE_TOKEN')

def chat(model_name: str, query: str, hf_token: str = hf_token) -> str:
    """
    Download Open-source models from Hugging Face and Send the query to the model.
    Optimized for maximum performance on multi-GPU HPC systems.
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

    # Configure quantization for memory efficiency
    # Use 4-bit quantization if model is large
    use_4bit = "70b" in model_name.lower() or "65b" in model_name.lower()
    
    if use_4bit:
        print("[INFO] Using 4-bit quantization for large model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    # Load tokenizer with caching
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir='./models', 
        token=hf_token,
        use_fast=True  # Use fast tokenizer implementation
    )
    
    # Load model with optimized settings
    print(f"[INFO] Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # bfloat16 for better numerical stability
        attn_implementation="flash_attention_2",
        cache_dir='./models',
        device_map="auto",  # Automatic sharding across GPUs
        token=hf_token,
        quantization_config=quantization_config,
        trust_remote_code=True,  # Required for some models
        use_cache=True,  # Enable KV caching for faster generation
        low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
    )
    
    # Set pad token ID if needed
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback
            tokenizer.pad_token_id = 0
            model.generation_config.pad_token_id = 0
    
    # Initialize DeepSpeed for inference
    ds_engine = deepspeed.init_inference(
        model=model,
        mp_size=torch.cuda.device_count(),
        dtype=torch.bfloat16,
        replace_method="auto",
        replace_with_kernel_inject=False
    )
    model = ds_engine.module
    
    # Print model device mapping information
    if hasattr(model, 'hf_device_map'):
        print(f"[INFO] Model distributed across devices: {model.hf_device_map}")
    
    print(f"\n[INFO] Using model: {model_name}")
    
    # Time Tracking
    start_time = time.time()
    
    # Create the chat template
    template = [
        {"role": "user", "content": query}
    ]
    
    # Apply the chat template correctly
    prompt = tokenizer.apply_chat_template(
        conversation=template,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text efficiently
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=4096  # Set appropriate context length
    )
    
    # Move inputs to the same device as the first model parameter
    first_param_device = next(model.parameters()).device
    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
    
    # Determine if we can use CUDA graphs for optimization
    use_cuda_graphs = torch.cuda.is_available() and hasattr(torch.cuda, 'is_current_stream_capturing') and \
                     not torch.cuda.is_current_stream_capturing() and \
                     torch.cuda.device_count() == 1  # Only for single GPU
    
    # Context manager for CUDA graphs if applicable
    cuda_graph_context = torch.cuda.graph(enabled=use_cuda_graphs) if use_cuda_graphs else nullcontext()
    
    # Generate outputs with optimized settings
    with cuda_graph_context:
        outputs = model.generate(
            **inputs, 
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            num_beams=1,  # Beam search is slower but can give better results
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Decode and print the output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up output if needed
    if '</think>' in output_text:
        output_text = output_text.split('</think>')[1]
    
    # Calculate tokens per second
    input_tokens = len(inputs['input_ids'][0])
    output_tokens = len(outputs[0]) - input_tokens
    tokens_per_second = output_tokens / elapsed_time
        
    print(f"\nResponse:\n{output_text}\n")
    print(f"[INFO] {model_name} took: {elapsed_time:.2f} seconds for response")
    print(f"[INFO] Generated {output_tokens} tokens at {tokens_per_second:.2f} tokens/sec\n")
    
    # Clean up to free memory
    del model
    del inputs
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return output_text
