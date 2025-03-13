"""
LLM factory for the Gmsh MCP system.
Provides a factory for creating LLM instances based on provider.
"""

from typing import Optional, Dict, Any

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .ollama_llm import OllamaLLM


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(
        provider: str, 
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance based on the provider.
        
        Args:
            provider: The LLM provider to use ('openai', 'anthropic', or 'ollama')
            model_name: The model name to use (provider-specific)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAILLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=kwargs.get("api_key")
            )
        elif provider == "anthropic":
            return AnthropicLLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=kwargs.get("api_key")
            )
        elif provider == "ollama":
            return OllamaLLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                host=kwargs.get("host")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 