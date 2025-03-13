"""
Base LLM interface for the Gmsh MCP system.
Defines the common interface for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.2, max_tokens: int = 2048):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: The name of the model to use
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to guide the LLM's behavior
            
        Returns:
            The generated response as a string
        """
        pass
    
    @abstractmethod
    async def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate code from the LLM, optimized for code generation.
        
        Args:
            prompt: The user prompt describing the code to generate
            system_prompt: Optional system prompt to guide the LLM's behavior
            
        Returns:
            The generated code as a string
        """
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a multi-turn conversation with the LLM.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        pass
    
    def extract_code_block(self, text: str) -> str:
        """
        Extract a Python code block from the LLM response.
        
        Args:
            text: The full text response from the LLM
            
        Returns:
            The extracted Python code, or the original text if no code block is found
        """
        # Look for Python code blocks (```python ... ```)
        if "```python" in text and "```" in text.split("```python", 1)[1]:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        
        # Look for generic code blocks (``` ... ```)
        if "```" in text and "```" in text.split("```", 1)[1]:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        
        return text 