# utils/gemini_client.py

import google.generativeai as genai
import os
from typing import Optional
import time
from dotenv import load_dotenv
load_dotenv()

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client with API key.
        API key can be passed directly or set as environment variable GEMINI_API_KEY
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize the free Gemini model
        self.model = genai.GenerativeModel('gemini-2.0-flash')  # Free tier model
        
    def chat(self, prompt: str, system: str = "", max_retries: int = 3) -> str:
        """
        Sends a prompt to Gemini and returns the generated response.
        
        Args:
            prompt: The user prompt/question
            system: System instructions (optional)
            max_retries: Number of retry attempts for rate limiting
            
        Returns:
            Generated response as string
        """
        try:
            # Combine system instructions with prompt if provided
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            
            # Make request with retries for rate limiting
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(full_prompt)
                    return response.text.strip()
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Handle rate limiting
                    if "quota" in error_msg or "rate" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                        else:
                            return f"ERROR: Rate limit exceeded after {max_retries} attempts"
                    
                    # Handle other API errors
                    elif "safety" in error_msg:
                        return "ERROR: Content blocked by safety filters"
                    else:
                        raise e
                        
        except Exception as e:
            return f"ERROR: {str(e)}"

# Convenience function for backward compatibility
def gemini_chat(prompt: str, model: str = "gemini-2.5-flash", system: str = "") -> str:
    """
    Convenience function that mimics the ollama_chat interface.
    
    Args:
        prompt: The user prompt/question
        model: Model name (defaults to free tier model)
        system: System instructions
        
    Returns:
        Generated response as string
    """
    try:
        client = GeminiClient()
        return client.chat(prompt, system)
    except Exception as e:
        return f"ERROR: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # Test the client
    import sys
    
    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        print("Get your free API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Test basic functionality
    client = GeminiClient()
    
    test_prompt = "Explain what machine learning is in simple terms."
    test_system = "You are a helpful AI assistant that explains complex topics clearly."
    
    print("Testing Gemini API connection...")
    response = client.chat(test_prompt, test_system)
    
    if response.startswith("ERROR:"):
        print(f"Failed: {response}")
    else:
        print("Success! Gemini API is working.")
        print(f"Response preview: {response[:100]}...")
        
    # Test convenience function
    print("\nTesting convenience function...")
    response2 = gemini_chat("What is 2+2?", system="Answer briefly.")
    print(f"Math test: {response2}")