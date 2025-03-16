#!/usr/bin/env python3
"""
News Depoliticizer Runner Script
--------------------------------
This script runs the entire news depoliticizer pipeline:
1. Fetches news from APIs
2. Processes it with a local LLM (preferring Ollama)
3. Starts the web server to display results
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def check_dependencies():
    """Check if required Python packages are installed"""
    try:
        import requests
        import pandas as pd
        import numpy as np
        import flask
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install requests pandas numpy flask python-dotenv scikit-learn")
        return False

def check_ollama():
    """Check if Ollama is installed and available"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout.strip()
            print(f"Ollama is available with models:\n{models}")
            return True
        else:
            print("Ollama is installed but returned an error")
            return False
    except FileNotFoundError:
        print("Ollama is not installed or not in PATH")
        return False

def run_news_aggregator(api_keys_available=True):
    """Run the news aggregator script"""
    print("\n1. Running News Aggregator...")
    
    if not api_keys_available:
        print("Warning: No API keys found in .env file. Using sample data for demonstration.")
        # Copy sample data if available, or create minimal sample
        sample_path = Path("sample_data/llm_input_data.json")
        if sample_path.exists():
            import shutil
            shutil.copy(sample_path, "llm_input_data.json")
            print("Using sample data for demonstration.")
            return True
        else:
            # Create minimal sample data
            import json
            sample_data = [
                {
                    "title": "Sample World News Story",
                    "content": "This is a sample news story for demonstration purposes. It contains some facts about world events that might typically be reported with political slant or inflammatory language.",
                    "sources": ["sample-news-source"],
                    "importance_rank": 1,
                    "total_sources": 1
                }
            ]
            with open("llm_input_data.json", "w") as f:
                json.dump(sample_data, f, indent=2)
            print("Created minimal sample data for demonstration.")
            return True
    
    try:
        result = subprocess.run([sys.executable, "news_aggregator.py"], check=True)
        return result.returncode == 0
    except subprocess.SubprocessError as e:
        print(f"Error running news aggregator: {e}")
        return False

def run_llm_processor(ollama_model="llama3"):
    """Run the LLM processor script with Ollama"""
    print("\n2. Running LLM Processor with Ollama...")
    
    # First check if we can directly call ollama to test it works
    try:
        test_result = subprocess.run(
            ["ollama", "run", ollama_model, "--help"], 
            capture_output=True, 
            text=True
        )
        if test_result.returncode != 0:
            print(f"Warning: Ollama test failed with: {test_result.stderr}")
            print("Continuing anyway, but you may encounter errors...")
    except Exception as e:
        print(f"Warning: Ollama test failed: {e}")
        print("Continuing anyway, but you may encounter errors...")
    
    try:
        # Run with verbosity to help debug issues
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output
        
        result = subprocess.run([
            sys.executable, 
            "llm_processor.py",
            "--ollama",
            "--ollama-model", ollama_model
        ], env=env, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print("LLM processing failed. Check the logs for details.")
            return False
    except subprocess.SubprocessError as e:
        print(f"Error running LLM processor: {e}")
        return False

def run_web_server(port=8080):
    """Run the web server"""
    print(f"\n3. Starting Web Server on port {port}...")
    try:
        # Start the server in a new process so it doesn't block
        web_server = subprocess.Popen([sys.executable, "web_display.py", "--port", str(port)])
        
        # Give the server time to start
        time.sleep(2)
        
        print("\n🎉 News Depoliticizer is now running!")
        print(f"📰 Open http://localhost:{port} in your browser to view the depoliticized news")
        print("Press Ctrl+C to stop the server")
        
        # Keep the process running until user interrupts
        try:
            web_server.wait()
        except KeyboardInterrupt:
            print("\nStopping web server...")
            web_server.terminate()
            web_server.wait(timeout=5)
        
        return True
    except Exception as e:
        print(f"Error running web server: {e}")
        return False

def check_api_keys():
    """Check if API keys are available in .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for at least one API key
        api_keys = [
            os.getenv("NEWSAPI_KEY"),
            os.getenv("NYT_KEY"),
            os.getenv("GUARDIAN_KEY")
        ]
        
        return any(key for key in api_keys)
    except ImportError:
        print("python-dotenv not installed, can't check for API keys")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the News Depoliticizer pipeline")
    parser.add_argument("--ollama-model", type=str, default="llama3",
                        help="Ollama model to use (default: llama3)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching news (use existing data)")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip processing with LLM (use existing processed data)")
    parser.add_argument("--web-only", action="store_true",
                        help="Only start the web server (skip fetching and processing)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run the web server on (default: 8080)")
    
    args = parser.parse_args()
    
    print("🗞️  News Depoliticizer 🤖")
    print("=======================")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check Ollama
    ollama_available = check_ollama()
    if not ollama_available:
        print("Warning: Ollama not found. The LLM processor may not work correctly.")
    
    # Check API keys
    api_keys_available = check_api_keys()
    if not api_keys_available:
        print("Warning: No API keys found in .env file.")
        print("You can get free API keys from:")
        print("- NewsAPI: https://newsapi.org/")
        print("- New York Times: https://developer.nytimes.com/")
        print("- The Guardian: https://open-platform.theguardian.com/")
        print("Add them to a .env file in the project directory.")
    
    # Run pipeline
    try:
        # Step 1: Fetch news (unless skipped)
        if args.web_only or args.skip_fetch:
            print("\n1. Skipping news fetching as requested.")
        else:
            success = run_news_aggregator(api_keys_available)
            if not success:
                print("Failed to fetch news. Check news_aggregator.py for errors.")
                return
        
        # Step 2: Process with LLM (unless skipped)
        if args.web_only or args.skip_process:
            print("\n2. Skipping LLM processing as requested.")
        else:
            success = run_llm_processor(args.ollama_model)
            if not success:
                print("Failed to process news with LLM. Check llm_processor.py for errors.")
                return
        
        # Step 3: Start web server
        run_web_server(port=args.port)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
