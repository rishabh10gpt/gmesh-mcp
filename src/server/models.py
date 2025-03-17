"""
Data models for the Gmsh MCP server API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from ..utils.config import OLLAMA_HOST, OLLAMA_MODEL

class LLMConfig(BaseModel):
    """LLM configuration model."""
    
    provider: str = Field(
        default="ollama",
        description="LLM provider to use ('openai', 'anthropic', or 'ollama')"
    )
    model_name: Optional[str] = Field(
        default=OLLAMA_MODEL,
        description="Model name to use (provider-specific)"
    )
    temperature: float = Field(
        default=0.2,
        description="Controls randomness in the response (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate",
        gt=0
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider (if not using environment variables)"
    )
    host: Optional[str] = Field(
        default=OLLAMA_HOST,
        description="Host URL for Ollama (if not using environment variables)"
    )


class MeshRequest(BaseModel):
    """Mesh generation request model."""
    
    prompt: str = Field(
        ...,
        description="Natural language description of the mesh to generate"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration (uses defaults if not provided)"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to guide the LLM's behavior"
    )


class MeshRefinementRequest(BaseModel):
    """Mesh refinement request model."""
    
    mesh_id: str = Field(
        ...,
        description="ID of the mesh to refine"
    )
    feedback: str = Field(
        ...,
        description="Natural language feedback on how to refine the mesh"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration (uses defaults if not provided)"
    )


class ChatRequest(BaseModel):
    """Chat request model."""
    
    message: str = Field(
        ...,
        description="User message"
    )
    mesh_id: Optional[str] = Field(
        default=None,
        description="ID of the mesh to associate with the conversation"
    )
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration (uses defaults if not provided)"
    )


class MeshStatistics(BaseModel):
    """Mesh statistics model."""
    
    num_nodes: Optional[int] = Field(
        default=None,
        description="Number of nodes in the mesh"
    )
    num_elements: Optional[int] = Field(
        default=None,
        description="Number of elements in the mesh"
    )
    element_types: Optional[List[str]] = Field(
        default=None,
        description="Types of elements in the mesh"
    )
    physical_groups: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Physical groups in the mesh"
    )
    bounding_box: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Bounding box of the mesh"
    )


class MeshResponse(BaseModel):
    """Mesh generation response model."""
    
    mesh_id: str = Field(
        ...,
        description="Unique identifier for the mesh"
    )
    statistics: MeshStatistics = Field(
        default_factory=MeshStatistics,
        description="Mesh statistics"
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Path to the output mesh file"
    )
    visualization_file: Optional[str] = Field(
        default=None,
        description="Path to the visualization file"
    )
    success: bool = Field(
        default=True,
        description="Whether the mesh generation was successful"
    )
    message: Optional[str] = Field(
        default=None,
        description="Message about the mesh generation"
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    
    response: str = Field(
        ...,
        description="LLM response"
    )
    mesh_id: Optional[str] = Field(
        default=None,
        description="ID of the mesh associated with the conversation"
    ) 