"""
Anthropic LLM provider for the Gmsh MCP system.
"""

import anthropic
from typing import Dict, List, Optional, Any

from ..utils.config import ANTHROPIC_API_KEY
from .base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider implementation."""
    
    def __init__(
        self, 
        model_name: str = "claude-3-opus-20240229", 
        temperature: float = 0.2, 
        max_tokens: int = 2048,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Anthropic LLM provider.
        
        Args:
            model_name: The Anthropic model to use (default: claude-3-opus-20240229)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            api_key: Anthropic API key (defaults to environment variable)
        """
        super().__init__(model_name, temperature, max_tokens)
        self.client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the Anthropic model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            The generated response as a string
        """
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    async def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate code from the Anthropic model, optimized for code generation.
        
        Args:
            prompt: The user prompt describing the code to generate
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            The generated code as a string
        """
        code_prompt = f"Generate Python code using the Gmsh API for the following task: {prompt}\n\nProvide ONLY the Python code without any explanations or markdown."
        
        if not system_prompt:
            system_prompt = "You are an expert in computational geometry and mesh generation using Gmsh. Generate clean, efficient Python code that uses the Gmsh API correctly."
        
        response = await self.generate(code_prompt, system_prompt)
        return self.extract_code_block(response)
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a multi-turn conversation with the Anthropic model.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System messages are handled separately in Anthropic
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Extract system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=anthropic_messages
        )
        
        return message.content[0].text 