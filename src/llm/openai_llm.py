"""
OpenAI LLM provider for the Gmsh MCP system.
"""

import openai
from typing import Dict, List, Optional, Any

from ..utils.config import OPENAI_API_KEY
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        temperature: float = 0.2, 
        max_tokens: int = 2048,
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI LLM provider.
        
        Args:
            model_name: The OpenAI model to use (default: gpt-4o)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            api_key: OpenAI API key (defaults to environment variable)
        """
        super().__init__(model_name, temperature, max_tokens)
        self.client = openai.OpenAI(api_key=api_key or OPENAI_API_KEY)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the OpenAI model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            The generated response as a string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def generate_code(self, prompt: str, feedback: Optional[str] = None) -> str:
        """
        Generate code from the OpenAI model, optimized for code generation.
        
        Args:
            prompt: The user prompt describing the code to generate
            feedback: Optional feedback from previous attempts
            
        Returns:
            The generated code as a string
        """
        # Use the standardized prompt template from the base class
        mesh_prompt = self.get_mesh_prompt_template(prompt, feedback)
        system_prompt = self.get_system_prompt()
        
        # Generate code using the OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mesh_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract and post-process the code
        code = response.choices[0].message.content
        return self.post_process_code(code)
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a multi-turn conversation with the OpenAI model.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content 