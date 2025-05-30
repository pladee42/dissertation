from pathlib import Path
import os
import json

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

def extract_json_from_markdown(text_response):
    start_delimiter = "```json\n"
    end_delimiter = "\n```"

    start_index = text_response.find(start_delimiter)
    end_index = text_response.rfind(end_delimiter)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Calculate the actual start of the JSON content
        json_start = start_index + len(start_delimiter)
        # Extract the raw JSON string
        json_string = text_response[json_start:end_index]
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("JSON block not found in the expected format.")
        return None