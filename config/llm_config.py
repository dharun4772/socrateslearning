# config/llm_config.py

from utils.ollama_client import ollama_chat
from utils.gemini_client import gemini_chat
from typing import Literal

# Available LLM providers
LLMProvider = Literal["ollama", "gemini"]

# Model configurations
OLLAMA_MODELS = {
    "llama3": "llama3",
    "llama3.1": "llama3.1", 
    "codellama": "codellama",
    "mistral": "mistral"
}

GEMINI_MODELS = {
    "gemini-2.0-flash": "gemini-2.0-flash",  # Free tier
    "gemini-2.0-pro": "gemini-2.0-pro",      # Paid tier (better quality)
    "gemini": "gemini-2.0-flash"             # Default alias
}

class LLMClient:
    """Unified client for different LLM providers"""
    
    def __init__(self, provider: LLMProvider = "gemini", model: str = "gemini-2.5-flash"):
        self.provider = provider
        self.model = model
        # Validate model for provider
        if provider == "ollama" and model not in OLLAMA_MODELS:
            raise ValueError(f"Model {model} not available for Ollama. Available: {list(OLLAMA_MODELS.keys())}")
        elif provider == "gemini" and model not in GEMINI_MODELS:
            raise ValueError(f"Model {model} not available for Gemini. Available: {list(GEMINI_MODELS.keys())}")
    
    def chat(self, prompt: str, system: str = "") -> str:
        """Send prompt to configured LLM provider"""
        
        if self.provider == "ollama":
            return ollama_chat(prompt, model=OLLAMA_MODELS[self.model], system=system)
        elif self.provider == "gemini":
            return gemini_chat(prompt, model=GEMINI_MODELS[self.model], system=system)
        else:
            return f"ERROR: Unknown provider {self.provider}"

# Convenience functions
def get_llm_client(provider: LLMProvider = "gemini", model: str = "gemini-2.5-flash") -> LLMClient:
    """Get configured LLM client"""
    return LLMClient(provider, model)

def chat_with_llm(prompt: str, system: str = "", provider: LLMProvider = "gemini", model: str = "gemini-2.5-flash") -> str:
    """Direct chat function with specified provider"""
    client = get_llm_client(provider, model)
    return client.chat(prompt, system)

# Example usage
if __name__ == "__main__":
    # Test different providers
    test_prompt = "What is the capital of France?"
    
    print("Testing Gemini (free):")
    gemini_response = chat_with_llm(test_prompt, provider="gemini", model="gemini-2.0-flash")
    print(f"Gemini: {gemini_response}\n")
    
    print("Testing Ollama:")
    ollama_response = chat_with_llm(test_prompt, provider="ollama", model="llama3")
    print(f"Ollama: {ollama_response}")