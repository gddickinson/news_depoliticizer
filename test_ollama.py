#!/usr/bin/env python3
"""
Simple script to test if Ollama is working correctly from Python
"""

import subprocess
import sys

def test_ollama_direct():
    """Test Ollama with a simple prompt using direct subprocess call"""
    print("Testing direct Ollama integration...")
    
    try:
        # First, check if Ollama is available
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("✓ Ollama is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("✗ Ollama is not installed or not in PATH")
        return False
    
    # List available models
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        print(f"Available models:\n{result.stdout}")
    except subprocess.SubprocessError:
        print("✗ Could not list Ollama models")
        return False
    
    # Test simple prompt
    try:
        cmd = ["ollama", "run", "llama3"]
        prompt = "Tell me a short joke about programming."
        
        print(f"Running: {' '.join(cmd)}")
        print(f"With prompt: {prompt}")
        
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            print("\nOutput from Ollama:")
            print("-" * 40)
            print(result.stdout)
            print("-" * 40)
            print("✓ Ollama responded successfully")
            return True
        else:
            print(f"✗ Ollama returned error code {result.returncode}")
            print(f"Error message: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running Ollama: {e}")
        return False

if __name__ == "__main__":
    if test_ollama_direct():
        print("\nOllama test passed! You should be able to use it with the news depoliticizer.")
        sys.exit(0)
    else:
        print("\nOllama test failed. Please check your Ollama installation.")
        sys.exit(1)
