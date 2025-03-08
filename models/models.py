from dotenv import load_dotenv
import os
import torch
import time
from vllm import LLM, SamplingParams

# Load environment variables once at module level
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

class ModelInference:
    def __init__(self, 
                 model_id: str, 
                 dtype: str = "bfloat16",
                 quantization: str = None):
        """Initialize the model once and keep it in memory"""
        
        print(f"Loading model {model_id}...")
        start_time = time.time()
        
        # Load the model once during initialization
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            download_dir='../downloaded_models',
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
            quantization=quantization,
            trust_remote_code=True
        )
        
        # Default sampling parameters
        self.default_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            max_tokens=4096
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    def format_prompt(self, 
                      query: str, 
                      model_name: str) -> str:
        """Format prompt based on model type"""
        
        model_lower = model_name.lower()
        
        # If it seems like a math or reasoning problem, add the boxed suggestion
        if "deepseek" in model_lower:
            if any(word in query.lower() for word in ["calculate", "solve", "math", "equation", "proof"]):
                return f"User: {query}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            else:
                return f"User: {query}"
        elif "llama" in model_lower:
            return f"<s>[INST] {query} [/INST]"
        elif "mistral" in model_lower:
            return f"<s>[INST] {query} [/INST]"
        elif "gemma" in model_lower:
            return f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"User: {query}\nAssistant:"
    
    def generate(self, 
                 query: str, 
                 model_name: str, 
                 custom_params: dict = None, 
                 remove_cot: bool = False) -> str:
        """Generate response for a given query"""
        prompt = self.format_prompt(query, model_name)
        params = custom_params if custom_params else self.default_params
        
        start_time = time.time()
        outputs = self.llm.generate(prompt, params)
        
        # Extract generated text
        output = outputs[0]
        try:
            output_text = output.outputs[0].text
        except (AttributeError, IndexError):
            try:
                output_text = output.outputs.text
            except AttributeError:
                output_text = str(output)
        
        gen_time = time.time() - start_time
        print(f"Generated response in {gen_time:.2f} seconds")
        
        # Remove the Chain of Thought part from DeepSeek's responses
        if remove_cot:
            output_text = output_text.split('</think>\n')[1]
        
        return output_text
