#!/usr/bin/env python3
"""Test script to verify GLIBCXX compatibility and environment setup"""

import os
import sys
import subprocess
import importlib.util

def check_glibc_version():
    """Check glibc version available on system"""
    try:
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
        print("=== GLIBC Version ===")
        print(result.stdout.split('\n')[0])
        return True
    except Exception as e:
        print(f"Error checking glibc version: {e}")
        return False

def check_glibcxx_versions():
    """Check available GLIBCXX versions"""
    try:
        libstdc_paths = [
            '/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
            '/usr/lib64/libstdc++.so.6',
            '/lib/x86_64-linux-gnu/libstdc++.so.6'
        ]
        
        for path in libstdc_paths:
            if os.path.exists(path):
                print(f"\n=== GLIBCXX versions in {path} ===")
                result = subprocess.run(['strings', path], capture_output=True, text=True)
                glibcxx_versions = [line for line in result.stdout.split('\n') if 'GLIBCXX' in line]
                for version in sorted(set(glibcxx_versions)):
                    print(version)
                return True
        
        print("No libstdc++.so.6 found in common locations")
        return False
    except Exception as e:
        print(f"Error checking GLIBCXX versions: {e}")
        return False

def test_cuda_availability():
    """Test CUDA functionality"""
    try:
        import torch
        print(f"\n=== CUDA Test ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        return True
    except Exception as e:
        print(f"CUDA test failed: {e}")
        return False

def test_vllm_import():
    """Test vLLM import"""
    try:
        import vllm
        print(f"\n=== vLLM Test ===")
        print("vLLM imported successfully")
        print(f"vLLM version: {vllm.__version__}")
        return True
    except Exception as e:
        print(f"vLLM import failed: {e}")
        return False

def test_transformers_import():
    """Test transformers import"""
    try:
        import transformers
        print(f"\n=== Transformers Test ===")
        print(f"Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"Transformers import failed: {e}")
        return False

def test_model_loading():
    """Test basic model loading"""
    try:
        from transformers import AutoTokenizer
        print(f"\n=== Model Loading Test ===")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Successfully loaded GPT-2 tokenizer")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Container Environment Compatibility Test ===")
    
    tests = [
        ("GLIBC Version", check_glibc_version),
        ("GLIBCXX Versions", check_glibcxx_versions),
        ("CUDA Availability", test_cuda_availability),
        ("vLLM Import", test_vllm_import),
        ("Transformers Import", test_transformers_import),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
    
    print("\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()