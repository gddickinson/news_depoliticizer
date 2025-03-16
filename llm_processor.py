import json
import os
import subprocess
from datetime import datetime
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("llm_processing.log"), logging.StreamHandler()]
)
logger = logging.getLogger("NewsLLM")

# Check if Ollama is available
def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        return False
    except FileNotFoundError:
        return False

USING_OLLAMA = check_ollama()
if USING_OLLAMA:
    logger.info("Ollama found and will be used as the primary LLM provider")

# Import LLM libraries - choose one based on your preferred local model
# Option 1: Using llama-cpp-python for Llama models
try:
    from llama_cpp import Llama
    USING_LLAMA = True
except ImportError:
    USING_LLAMA = False
    if not USING_OLLAMA:
        logger.warning("llama-cpp-python not found, trying transformers instead")

# Option 2: Using Hugging Face transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    USING_TRANSFORMERS = True
except ImportError:
    USING_TRANSFORMERS = False
    if not USING_OLLAMA and not USING_LLAMA:
        logger.error("Neither Ollama, llama-cpp-python, nor transformers found. Please install one of them.")
        raise ImportError("No LLM library available")


class NewsLLMProcessor:
    def __init__(self, model_path=None, use_ollama=True, ollama_model="llama3"):
        """
        Initialize the News LLM Processor
        
        Args:
            model_path: Path to the model. If None, will use default models
            use_ollama: Whether to use Ollama if available
            ollama_model: Which Ollama model to use (default: llama3)
        """
        self.model_path = model_path
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.llm = None
        self.tokenizer = None
        self.llm_type = None
        
        # Load the appropriate model
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate LLM model based on available libraries"""
        # First try Ollama if enabled and available
        if self.use_ollama and USING_OLLAMA:
            # Check if the specified model exists in Ollama
            try:
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True
                )
                if self.ollama_model in result.stdout:
                    logger.info(f"Using Ollama with model: {self.ollama_model}")
                    self.llm_type = "ollama"
                    return
                else:
                    available_models = result.stdout.strip()
                    logger.warning(f"Requested Ollama model '{self.ollama_model}' not found. Available models:\n{available_models}")
                    logger.warning("Falling back to other LLM options")
            except Exception as e:
                logger.warning(f"Error checking Ollama models: {str(e)}")
        
        # Next, try llama.cpp if available
        if USING_LLAMA:
            # Default to a smaller Llama model if path not specified
            model_path = self.model_path or "models/llama-3-8b-instruct.gguf"
            
            logger.info(f"Loading Llama model from {model_path}")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,  # Context window size
                    n_gpu_layers=-1  # Use GPU if available
                )
                self.llm_type = "llama.cpp"
                logger.info("Llama model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Error loading Llama model: {str(e)}")
                # Continue to try next option instead of raising
        
        # Finally, try transformers if available        
        if USING_TRANSFORMERS:
            # Default to Mistral 7B if path not specified
            model_path = self.model_path or "mistralai/Mistral-7B-Instruct-v0.2"
            
            logger.info(f"Loading transformers model {model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Check for GPU availability
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Load in 8-bit precision for memory efficiency if on GPU
                load_in_8bit = device == "cuda"
                
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    load_in_8bit=load_in_8bit,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                
                self.llm_type = "transformers"
                logger.info("Transformers model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Error loading transformers model: {str(e)}")
                # Continue to next option
        
        # If we get here, no LLM could be loaded
        raise ValueError("No LLM library or model available")
    
    def generate_response(self, prompt, max_tokens=1024, temperature=0.7):
        """Generate a response from the loaded LLM"""
        if self.llm_type == "ollama":
            try:
                # Use a more direct approach - pipe the prompt to ollama
                cmd = ["ollama", "run", self.ollama_model]
                
                logger.info(f"Generating response with Ollama model {self.ollama_model}")
                
                # Run ollama with the prompt as input
                result = subprocess.run(
                    cmd, 
                    input=prompt, 
                    text=True,
                    capture_output=True
                )
                    
                if result.returncode == 0:
                    # Extract response (skip the prompt part)
                    response = result.stdout.strip()
                    
                    # Process to remove any echoed prompt
                    # Find end of the prompt marker and extract content after it
                    response_marker = "<rewritten_article>"
                    if response_marker in response.lower():
                        # Extract content after marker
                        response = response.split(response_marker, 1)[1].strip()
                    elif prompt.strip() in response:
                        # If no marker but prompt is echoed, remove it
                        response = response[len(prompt.strip()):].strip()
                    
                    return response
                else:
                    logger.error(f"Ollama error: {result.stderr}")
                    return f"Error generating response: {result.stderr}"
            except Exception as e:
                logger.error(f"Error generating with Ollama: {str(e)}")
                return "Error generating response with Ollama"
                
        elif self.llm_type == "llama.cpp":
            try:
                output = self.llm(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["</ARTICLE>", "</REWRITTEN_ARTICLE>"]  # Custom stop tokens
                )
                return output['choices'][0]['text'].strip()
            except Exception as e:
                logger.error(f"Error generating with Llama: {str(e)}")
                return "Error generating response"
                
        elif self.llm_type == "transformers":
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = inputs.to(self.llm.device)
                
                # Generate
                generate_ids = self.llm.generate(
                    inputs.input_ids, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and return text after the prompt
                output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                # Remove the prompt from the output
                output = output[len(prompt):]
                
                return output.strip()
            except Exception as e:
                logger.error(f"Error generating with transformers: {str(e)}")
                return "Error generating response"
        
        return "No LLM model available to generate response"
    
    def create_depoliticized_prompt(self, news_item, temperature=0.7):
        """Create a prompt for depoliticizing a news article"""
        
        # For Ollama, we can add parameters directly in the prompt if needed
        if self.llm_type == "ollama":
            # Include model parameters in the prompt for Ollama
            return f"""
<task>
You are a neutral news editor. Rewrite the following news story to remove political bias, 
inflammatory language, and partisan framing while preserving all key facts.

<article_info>
Title: {news_item['title']}
Importance Rank: {news_item['importance_rank']}
Number of Sources: {news_item['total_sources']}
</article_info>

<original_text>
{news_item['content'][:3000]}  # Truncate if too long
</original_text>

<instructions>
1. Maintain all factual information and key details.
2. Remove partisan language, loaded terms, and politically charged framing.
3. Present multiple perspectives where relevant.
4. Use neutral, objective language.
5. Structure the article with the most important information first.
6. Keep approximately the same length as the original.
</instructions>

<rewritten_article>
"""
        # Standard prompt for other LLM types
        else:
            return f"""<TASK>
You are a neutral news editor. Rewrite the following news story to remove political bias, 
inflammatory language, and partisan framing while preserving all key facts.

<ARTICLE_INFO>
Title: {news_item['title']}
Importance Rank: {news_item['importance_rank']}
Number of Sources: {news_item['total_sources']}
</ARTICLE_INFO>

<ORIGINAL_TEXT>
{news_item['content'][:3000]}  # Truncate if too long
</ORIGINAL_TEXT>

<INSTRUCTIONS>
1. Maintain all factual information and key details.
2. Remove partisan language, loaded terms, and politically charged framing.
3. Present multiple perspectives where relevant.
4. Use neutral, objective language.
5. Structure the article with the most important information first.
6. Keep approximately the same length as the original.

Respond with only the rewritten article text.
</TASK>

<REWRITTEN_ARTICLE>
"""
    
    def process_news_data(self, input_file="llm_input_data.json", output_dir="processed_news"):
        """Process all news items and rewrite them using the LLM"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Current timestamp for the output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/depoliticized_news_{timestamp}.json"
        
        try:
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            logger.info(f"Loaded {len(news_data)} news items for processing")
            
            # Process each news item
            processed_items = []
            
            for i, item in enumerate(news_data):
                logger.info(f"Processing item {i+1}/{len(news_data)}: {item['title']}")
                
                # Create prompt with temperature parameter
                prompt = self.create_depoliticized_prompt(item, temperature=0.7)
                
                # Generate rewritten article
                rewritten = self.generate_response(prompt)
                
                # Save processed item
                processed_item = {
                    'original_title': item['title'],
                    'importance_rank': item['importance_rank'],
                    'sources': item['sources'],
                    'original_content': item['content'][:500] + "...",  # Truncated for saving space
                    'rewritten_content': rewritten,
                    'processed_at': datetime.now().isoformat()
                }
                
                processed_items.append(processed_item)
                
                # Short delay to avoid overloading the LLM (especially important for local models)
                time.sleep(1)
            
            # Save all processed items
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_items, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Successfully processed {len(processed_items)} news items and saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing news data: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        import argparse
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Process news data with a local LLM")
        parser.add_argument("--ollama", action="store_true", default=True, 
                           help="Use Ollama if available (default: True)")
        parser.add_argument("--ollama-model", type=str, default="llama3",
                           help="Ollama model to use (default: llama3)")
        parser.add_argument("--model-path", type=str, default=None,
                           help="Path to a local model file (for llama.cpp) or model name (for transformers)")
        parser.add_argument("--input", type=str, default="llm_input_data.json",
                           help="Input JSON file with news data (default: llm_input_data.json)")
        parser.add_argument("--output-dir", type=str, default="processed_news",
                           help="Directory to save processed news (default: processed_news)")
        
        args = parser.parse_args()
        
        # Check if Ollama is available and preferred
        if args.ollama and USING_OLLAMA:
            print(f"Using Ollama with model: {args.ollama_model}")
            processor = NewsLLMProcessor(
                use_ollama=True, 
                ollama_model=args.ollama_model
            )
        else:
            # Fall back to other models
            print(f"Using model path: {args.model_path}")
            processor = NewsLLMProcessor(
                model_path=args.model_path,
                use_ollama=False
            )
        
        # Process news data
        output_file = processor.process_news_data(
            input_file=args.input,
            output_dir=args.output_dir
        )
        
        print(f"Processing complete. Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
