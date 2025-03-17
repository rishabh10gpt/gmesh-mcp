"""
Controller for interacting with LLM providers.
"""

import os
from typing import Optional, List, Dict, Any
from ..llm.ollama_llm import OllamaLLM
from ..llm.anthropic_llm import AnthropicLLM
from ..llm.openai_llm import OpenAILLM

class LLMController:
    """Controller for interacting with LLM providers."""
    
    SUPPORTED_PROVIDERS = ["ollama", "anthropic", "openai"]
    DEFAULT_PROVIDER = "ollama"
    
    def __init__(self, llm_provider: str = DEFAULT_PROVIDER, model_name: Optional[str] = None):
        """
        Initialize the LLM controller.
        
        Args:
            llm_provider: LLM provider to use (ollama, anthropic, openai)
            model_name: Model name to use (provider-specific)
        """
        if llm_provider not in self.SUPPORTED_PROVIDERS:
            llm_provider = self.DEFAULT_PROVIDER
            print(f"Warning: Unsupported provider '{llm_provider}'. Using default provider '{self.DEFAULT_PROVIDER}'")
            
        self.provider = llm_provider
        self.model = model_name
        self._setup_provider()
    
    def _setup_provider(self) -> None:
        """Set up the LLM provider based on the configuration."""
        if self.provider == "ollama":
            if not os.getenv("OLLAMA_HOST"):
                raise ValueError("OLLAMA_HOST environment variable is not set")
            self.llm = OllamaLLM(model_name=self.model)
        elif self.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            self.llm = AnthropicLLM(model_name=self.model)
        elif self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.llm = OpenAILLM(model_name=self.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        print(f"Using provider: {self.provider}")
        print(f"Using model: {self.model or self.llm.model_name}")
    
    async def generate_code(self, prompt: str, feedback: Optional[str] = None) -> str:
        """
        Generate Gmsh code based on the prompt and feedback.
        
        Args:
            prompt: Natural language prompt describing the mesh to generate
            feedback: Optional feedback from previous generation attempts
            
        Returns:
            Generated Gmsh code as a string
        """
        # Pass the prompt and feedback directly to the LLM implementation
        # Each implementation now uses the standardized prompt template
        return await self.llm.generate_code(prompt, feedback)
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a multi-turn conversation with the LLM.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        return await self.llm.chat(messages)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to guide the LLM's behavior
            
        Returns:
            The generated response as a string
        """
        return await self.llm.generate(prompt, system_prompt) 