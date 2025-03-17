"""
Gmsh MCP (Model context Protocol) package.
"""

from .core import MeshGenerator, GmshController, MeshData
from .llm import LLMFactory, BaseLLM, OpenAILLM, AnthropicLLM, OllamaLLM

__version__ = "0.1.0"

__all__ = [
    "MeshGenerator",
    "GmshController",
    "MeshData",
    "LLMFactory",
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM"
] 