"""
Ollama LLM provider for the Gmsh MCP system.
Provides support for local LLMs using Ollama.
"""

import json
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
import subprocess
import re
import textwrap
import os
import requests

from ..utils.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_API_BASE_URL, OLLAMA_MODEL_NAME
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation for local LLMs."""
    
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME):
        """
        Initialize the Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        super().__init__()  # Call parent's __init__ without arguments
        self.host = OLLAMA_HOST
        self.api_base = OLLAMA_API_BASE_URL
        self.model = model_name  # Store model name for API calls
        self.base_url = OLLAMA_API_BASE_URL
    
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Ollama API.
        
        Args:
            endpoint: API endpoint to call
            data: Request data
            
        Returns:
            Response data as a dictionary
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/{endpoint}",
                json=data
            ) as response:
                error_text = await response.text()
                if response.status != 200:
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
            "model": self.model,
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
    
    async def generate_code(self, prompt: str, feedback: Optional[str] = None) -> str:
        """
        Generate Gmsh code based on the given prompt.
        
        Args:
            prompt: Natural language prompt describing the mesh to generate
            feedback: Optional feedback from previous attempts
            
        Returns:
            Generated Gmsh code as a string
        """
        # Use the standardized prompt template from the base class
        prompt_with_settings = self.get_mesh_prompt_template(prompt, feedback)
        
        # Generate code using the LLM
        response = await self._make_request("generate", {
            "model": self.model,
            "prompt": prompt_with_settings,
            "stream": False
        })
        
        # Extract code from response
        code = response["response"].strip()
        
        # Post-process the code
        return self.post_process_code(code)
    
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
            "model": self.model,
            "messages": formatted_messages,
            "options": {
                "temperature": 0.2,
                "num_predict": 2048
            }
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = await self._make_request("chat", payload)
        return response.get("message", {}).get("content", "")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are an expert in generating Gmsh code for mesh generation.
Your task is to generate Python code that uses the Gmsh API to create meshes based on natural language descriptions.

Rules for code generation:
1. Keep each line under 80 characters
2. Use descriptive variable names
3. Include proper error handling
4. Follow Gmsh's Python API conventions
5. Use the correct function signatures:
   - For boxes: gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
   - For cylinders: gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r)
   - For spheres: gmsh.model.occ.addSphere(x, y, z, r)
   - For boolean operations:
     * gmsh.model.occ.cut([(3, target)], [(3, tool)])
     * gmsh.model.occ.fuse([(3, obj1)], [(3, obj2)])
     * gmsh.model.occ.intersect([(3, obj1)], [(3, obj2)])
6. Always include proper mesh quality settings
7. Use appropriate mesh size settings for different regions
8. Include proper synchronization calls
9. Handle boolean operations correctly
10. Clean up resources properly

The code should:
1. Initialize Gmsh
2. Create the geometry using the occ module (not geo)
3. Set mesh parameters
4. Generate the mesh
5. Save the mesh
6. Finalize Gmsh

Example box creation:
```python
# Create a box
box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)  # x=0, y=0, z=0, dx=10, dy=10, dz=10
```

Example cylinder creation:
```python
# Create a cylinder along the z-axis
cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 10, 1)  # x=0, y=0, z=0, dx=0, dy=0, dz=10, r=1
```

Example boolean operation:
```python
# Cut cylinder from box
gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
gmsh.model.occ.synchronize()  # Always synchronize after boolean operations
```

Example mesh settings:
```python
# Set mesh algorithm
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1 = Netgen
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.option.setNumber("Mesh.Optimize", 1)

# Set mesh size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1)
```

Remember to:
1. Use proper error handling with try/except
2. Include mesh quality settings
3. Set appropriate mesh sizes
4. Handle boolean operations correctly
5. Always call gmsh.model.occ.synchronize() after geometry operations
6. Clean up resources with gmsh.finalize()
7. Use the occ module instead of geo for all geometry operations"""

    def _add_mesh_quality_settings(self, prompt: str) -> str:
        """Add mesh quality settings to the prompt."""
        return f"""{prompt}

IMPORTANT MESH QUALITY SETTINGS:
1. Set mesh algorithm to Netgen (Algorithm3D = 1)
2. Enable mesh optimization (Optimize = 1)
3. Enable Netgen optimization (OptimizeNetgen = 1)
4. Set appropriate mesh size based on model dimensions (typically 1/10th of smallest dimension)
5. Add these settings BEFORE mesh generation:
   gmsh.option.setNumber("Mesh.Algorithm3D", 1)
   gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
   gmsh.option.setNumber("Mesh.Optimize", 1)
   mesh_size = model_size / 10  # Calculate based on model dimensions
   gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)""" 