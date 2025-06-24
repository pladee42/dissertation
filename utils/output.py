from pathlib import Path
import os
import json
from pathlib import Path
import json
from typing import Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError
from models.schemas import Checklist, JudgmentResult


def open_response_files(response_file: str):
    """Open all response files if response_file == None"""
    
    response_dir = "./output/responses"
    response_dict = {}
    
    if not Path(response_dir).exists():
        print(f"Error: Folder '{response_dir}' does not exist.")
    else:
        # Open only 1 given file
        if response_file:
            response_path = Path(response_dir) / response_file
            filename = response_path.name.removesuffix('.txt')
            print(f"\n--- Opening and reading: {response_path.name} ---")
            with open(response_path, 'r', encoding='utf-8') as f:
                response_dict[filename] = f.read()
        
        # Open All Files
        else:
            response_path = Path(response_dir)
            for file_path in response_path.glob('*.txt'):
                filename = file_path.name.removesuffix('.txt')
                print(f"\n--- Opening and reading: {file_path.name} ---")
                with open(file_path, 'r', encoding='utf-8') as f:
                    response_dict[filename] = f.read()
                
    return response_dict

T = TypeVar('T', bound=BaseModel)

def extract_and_validate_json(text_response: str, model_class: Type[T]) -> T:
    """Extract JSON from markdown and validate against Pydantic model"""
    
    # Try to extract from markdown first
    json_data = extract_json_from_markdown(text_response)
    
    # If markdown extraction fails, try direct JSON parsing
    if json_data is None:
        try:
            json_data = json.loads(text_response.strip())
        except json.JSONDecodeError:
            # Try to find JSON in the text
            json_data = _find_json_in_text(text_response)
    
    if json_data is None:
        raise ValueError("No valid JSON found in response")
    
    # Validate against Pydantic model
    try:
        return model_class(**json_data)
    except ValidationError as e:
        raise ValueError(f"JSON validation failed: {e}")

def _find_json_in_text(text: str) -> Optional[dict]:
    """Try to find JSON object in text"""
    import re
    
    # Look for JSON-like patterns
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

# Your existing extract_json_from_markdown function remains the same
def extract_json_from_markdown(text_response: str) -> Optional[dict]:
    """Extract JSON from markdown code blocks"""
    start_delimiter = "```"
    end_delimiter = "\n```"
    
    start_index = text_response.find(start_delimiter)
    end_index = text_response.rfind(end_delimiter)
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_start = start_index + len(start_delimiter)
        json_string = text_response[json_start:end_index]
        
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    
    return None