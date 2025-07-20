#!/usr/bin/env python3
"""
Model Manager for DPO Training
Handles model availability checking and downloading
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model availability and downloading for DPO training"""
    
    def __init__(self, cache_dir: str = "../downloaded_models"):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(exist_ok=True)
        
        # Set HuggingFace cache to use our directory
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def load_model_registry(self) -> Dict:
        """Load model registry configuration"""
        with open('configs/model_registry.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_cache_path(self, model_id: str) -> Path:
        """Get expected cache path for a model"""
        # HuggingFace cache format: models--{org}--{model}
        safe_model_id = model_id.replace("/", "--")
        return self.cache_dir / f"models--{safe_model_id}"
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is already downloaded"""
        cache_path = self.get_model_cache_path(model_id)
        
        # Check if cache directory exists and has model files
        if not cache_path.exists():
            return False
        
        # Look for essential model files
        essential_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        alternative_files = ["model.safetensors", "pytorch_model-00001-of-*.bin"]
        
        has_config = (cache_path / "config.json").exists()
        has_model = any([
            (cache_path / f).exists() for f in essential_files[1:]
        ]) or any([
            list(cache_path.glob(pattern)) for pattern in alternative_files
        ])
        
        return has_config and has_model
    
    def check_model_availability(self, model_keys: List[str]) -> Dict[str, Dict]:
        """Check availability of multiple models"""
        registry = self.load_model_registry()
        results = {}
        
        for model_key in model_keys:
            if model_key not in registry['models']:
                results[model_key] = {
                    'status': 'unknown',
                    'error': f'Model {model_key} not found in registry'
                }
                continue
            
            model_info = registry['models'][model_key]
            
            # Load model config to get the actual model_id
            try:
                with open(model_info['config_file'], 'r') as f:
                    config = yaml.safe_load(f)
                model_id = config['model']['base_model']
            except Exception as e:
                results[model_key] = {
                    'status': 'error',
                    'error': f'Failed to load config: {e}'
                }
                continue
            
            # Check if model is cached
            is_cached = self.is_model_cached(model_id)
            cache_path = self.get_model_cache_path(model_id)
            
            results[model_key] = {
                'status': 'cached' if is_cached else 'missing',
                'model_id': model_id,
                'cache_path': str(cache_path),
                'size': model_info['size'],
                'memory_requirement': model_info['memory_requirement']
            }
        
        return results
    
    def download_model(self, model_id: str, force: bool = False) -> bool:
        """Download a model if not already cached"""
        if not force and self.is_model_cached(model_id):
            logger.info(f"Model {model_id} already cached, skipping download")
            return True
        
        try:
            logger.info(f"Downloading model: {model_id}")
            
            # Download tokenizer first (smaller, quick validation)
            logger.info(f"Downloading tokenizer for {model_id}...")
            
            # Handle SentencePiece tokenizer models (like Vicuna)
            tokenizer_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True
            }
            
            # Use slow tokenizer for models with SentencePiece issues
            if 'vicuna' in model_id.lower() or 'lmsys' in model_id.lower():
                tokenizer_kwargs["use_fast"] = False
                logger.info(f"Using slow tokenizer for {model_id} to handle SentencePiece conversion")
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
            
            # Download model weights WITHOUT loading into memory
            # Use snapshot_download for memory-efficient downloading
            logger.info(f"Downloading model weights for {model_id}...")
            from huggingface_hub import snapshot_download
            
            # Download model files to cache without instantiating
            cache_path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                repo_type="model",
                local_files_only=False
            )
            
            # Clean up tokenizer reference
            del tokenizer
            
            logger.info(f"Successfully downloaded {model_id} to {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            return False
    
    def ensure_models_available(self, model_keys: List[str]) -> Tuple[List[str], List[str]]:
        """Ensure all models are available, download if needed"""
        availability = self.check_model_availability(model_keys)
        
        available_models = []
        failed_models = []
        
        for model_key, info in availability.items():
            if info['status'] == 'cached':
                available_models.append(model_key)
                logger.info(f"✅ {model_key}: Already cached")
            
            elif info['status'] == 'missing':
                model_id = info['model_id']
                logger.info(f"⬇️ {model_key}: Downloading {model_id}...")
                
                if self.download_model(model_id):
                    available_models.append(model_key)
                    logger.info(f"✅ {model_key}: Downloaded successfully")
                else:
                    failed_models.append(model_key)
                    logger.error(f"❌ {model_key}: Download failed")
            
            else:  # error or unknown
                failed_models.append(model_key)
                logger.error(f"❌ {model_key}: {info.get('error', 'Unknown error')}")
        
        return available_models, failed_models
    
    def get_cache_stats(self) -> Dict:
        """Get cache directory statistics"""
        if not self.cache_dir.exists():
            return {'total_size': 0, 'model_count': 0, 'models': []}
        
        models = []
        total_size = 0
        
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name.startswith('models--'):
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                models.append({
                    'name': item.name.replace('models--', '').replace('--', '/'),
                    'size_mb': round(size / (1024 * 1024), 2),
                    'path': str(item)
                })
                total_size += size
        
        return {
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
            'model_count': len(models),
            'models': sorted(models, key=lambda x: x['size_mb'], reverse=True)
        }

def main():
    """CLI interface for model management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage models for DPO training')
    parser.add_argument('--check', nargs='+', help='Check availability of specific models')
    parser.add_argument('--download', nargs='+', help='Download specific models')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--cache-dir', default='../downloaded_models', help='Cache directory path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    manager = ModelManager(args.cache_dir)
    
    if args.stats:
        stats = manager.get_cache_stats()
        print(f"\n📊 Model Cache Statistics:")
        print(f"Total size: {stats['total_size_gb']} GB")
        print(f"Model count: {stats['model_count']}")
        
        if stats['models']:
            print(f"\nCached models:")
            for model in stats['models']:
                print(f"  {model['name']}: {model['size_mb']} MB")
    
    if args.check:
        print(f"\n🔍 Checking model availability...")
        availability = manager.check_model_availability(args.check)
        
        for model_key, info in availability.items():
            status_emoji = "✅" if info['status'] == 'cached' else "❌"
            print(f"  {status_emoji} {model_key}: {info['status']}")
            if info['status'] == 'missing':
                print(f"    Model ID: {info['model_id']}")
    
    if args.download:
        print(f"\n⬇️ Ensuring models are available...")
        available, failed = manager.ensure_models_available(args.download)
        
        print(f"\nResults:")
        print(f"  Available: {len(available)} models")
        print(f"  Failed: {len(failed)} models")
        
        if failed:
            print(f"  Failed models: {', '.join(failed)}")

if __name__ == "__main__":
    main()