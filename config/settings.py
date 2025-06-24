from pydantic_settings import BaseSettings
from typing import Dict, Optional

class Settings(BaseSettings):
    huggingface_token: Optional[str] = None
    download_dir: str = "./downloaded_models"
    output_dir: str = "./output"
    max_retries: int = 3
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    default_max_tokens: int = 4096
    default_repetition_penalty: float = 1.2
    
    class Config:
        env_file = ".env"

settings = Settings()
