"""
Mesh generator for the MCP system.
Integrates LLM and Gmsh controller to generate meshes from natural language prompts.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union

from ..llm.factory import LLMFactory
from ..llm.base import BaseLLM
from .gmsh_controller import GmshController, MeshData
from ..utils.config import SYSTEM_PROMPT, OUTPUT_DIR


class MeshGenerator:
    """
    Mesh generator that uses LLMs to generate Gmsh code from natural language prompts.
    """
    
    def __init__(
        self, 
        llm_provider: str = "openai", 
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs
    ):
        """
        Initialize the mesh generator.
        
        Args:
            llm_provider: The LLM provider to use ('openai', 'anthropic', or 'ollama')
            model_name: The model name to use (provider-specific)
            temperature: Controls randomness in the response (0.0 to 1.0)
            **kwargs: Additional provider-specific arguments
        """
        self.llm = LLMFactory.create_llm(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        self.gmsh_controller = GmshController()
        self.conversation_history = []
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> MeshData:
        """
        Generate a mesh from a natural language prompt.
        
        Args:
            prompt: Natural language description of the mesh to generate
            system_prompt: Optional system prompt to guide the LLM's behavior
            
        Returns:
            Generated mesh data
        """
        # Generate Gmsh code from the prompt
        code = await self.llm.generate_code(prompt, system_prompt or SYSTEM_PROMPT)
        
        # Execute the code
        success, output, mesh_data = self.gmsh_controller.execute_code(code)
        
        # Store the conversation
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Generated code:\n```python\n{code}\n```\n\nExecution {'succeeded' if success else 'failed'}:\n{output}"
        })
        
        # Save the conversation history
        self._save_conversation(mesh_data.mesh_id)
        
        return mesh_data
    
    async def refine(self, mesh_data: MeshData, feedback: str) -> MeshData:
        """
        Refine a mesh based on feedback.
        
        Args:
            mesh_data: Mesh data to refine
            feedback: Natural language feedback on how to refine the mesh
            
        Returns:
            Refined mesh data
        """
        # Add the feedback to the conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"Refine the mesh with the following feedback: {feedback}"
        })
        
        # Generate a system prompt with mesh statistics
        stats_prompt = f"""
You are refining a mesh with the following statistics:
- Number of nodes: {mesh_data.statistics.get('num_nodes', 'unknown')}
- Number of elements: {mesh_data.statistics.get('num_elements', 'unknown')}
- Element types: {', '.join(mesh_data.statistics.get('element_types', ['unknown']))}

The user wants to refine the mesh with this feedback: {feedback}

Generate Gmsh Python code that loads the existing mesh and refines it according to the feedback.
"""
        
        # Generate Gmsh code for refinement
        code = await self.llm.generate_code(feedback, stats_prompt)
        
        # Execute the code
        success, output, new_mesh_data = self.gmsh_controller.execute_code(code)
        
        # Store the conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": f"Generated refinement code:\n```python\n{code}\n```\n\nExecution {'succeeded' if success else 'failed'}:\n{output}"
        })
        
        # Save the conversation history
        self._save_conversation(new_mesh_data.mesh_id)
        
        return new_mesh_data
    
    async def chat(self, message: str) -> str:
        """
        Chat with the LLM about mesh generation.
        
        Args:
            message: User message
            
        Returns:
            LLM response
        """
        # Add the message to the conversation
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Get a response from the LLM
        response = await self.llm.chat(self.conversation_history)
        
        # Add the response to the conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _save_conversation(self, mesh_id: str) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            mesh_id: Mesh ID to associate with the conversation
        """
        conversation_file = os.path.join(OUTPUT_DIR, f"{mesh_id}_conversation.json")
        with open(conversation_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_mesh(self, filepath: str) -> MeshData:
        """
        Load a mesh from a file.
        
        Args:
            filepath: Path to the mesh file
            
        Returns:
            Loaded mesh data
        """
        return self.gmsh_controller.load_mesh(filepath)
    
    def export_mesh(self, mesh_data: MeshData, format: str = "msh") -> str:
        """
        Export a mesh to a file in the specified format.
        
        Args:
            mesh_data: Mesh data to export
            format: Export format (msh, vtk, etc.)
            
        Returns:
            Path to the exported file
        """
        return self.gmsh_controller.export_mesh(mesh_data, format) 