"""
Ollama LLM provider for the Gmsh MCP system.
Provides support for local LLMs using Ollama.
"""

import json
import aiohttp
from typing import Dict, List, Optional, Any

from ..utils.config import OLLAMA_HOST, OLLAMA_MODEL
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama LLM provider implementation for local LLMs."""
    
    def __init__(
        self, 
        model_name: str = None, 
        temperature: float = 0.2, 
        max_tokens: int = 2048,
        host: str = None
    ):
        """
        Initialize the Ollama LLM provider.
        
        Args:
            model_name: The Ollama model to use (defaults to config)
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            host: Ollama host URL (defaults to environment variable)
        """
        super().__init__(model_name or OLLAMA_MODEL, temperature, max_tokens)
        self.host = host or OLLAMA_HOST
        self.api_base = f"{self.host.rstrip('/')}/api"
    
    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Ollama API.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.api_base}/{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error ({response.status}): {error_text}")
                
                return await response.json()
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            The generated response as a string
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = await self._make_request("generate", payload)
        return response.get("response", "")
    
    async def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate code from the Ollama model, optimized for code generation.
        
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
        Have a multi-turn conversation with the Ollama model.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        # Extract system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Format messages for Ollama chat
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System message is handled separately
            
            formatted_messages.append({
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"]
            })
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = await self._make_request("chat", payload)
        return response.get("message", {}).get("content", "") 