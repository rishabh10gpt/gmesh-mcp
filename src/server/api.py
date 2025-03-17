"""
FastAPI server for the Gmsh MCP system.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ..core.mesh_generator import MeshGenerator
from ..core.gmsh_controller import MeshData
from ..utils.config import OUTPUT_DIR
from .models import (
    MeshRequest, 
    MeshRefinementRequest, 
    ChatRequest, 
    MeshResponse, 
    ChatResponse,
    MeshStatistics
)


# Create the FastAPI app
app = FastAPI(
    title="Gmsh MCP API",
    description="API for generating meshes using natural language prompts",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store active mesh generators
mesh_generators: Dict[str, MeshGenerator] = {}


def get_mesh_generator(mesh_id: Optional[str] = None) -> MeshGenerator:
    """
    Get or create a mesh generator.
    
    Args:
        mesh_id: ID of the mesh to get the generator for
        
    Returns:
        Mesh generator instance
    """
    if mesh_id and mesh_id in mesh_generators:
        return mesh_generators[mesh_id]
    
    # Create a new mesh generator with default settings
    generator = MeshGenerator()
    
    if mesh_id:
        mesh_generators[mesh_id] = generator
    
    return generator


def mesh_data_to_response(mesh_data: MeshData, success: bool = True, message: Optional[str] = None) -> MeshResponse:
    """
    Convert mesh data to a response model.
    
    Args:
        mesh_data: Mesh data to convert
        success: Whether the operation was successful
        message: Optional message about the operation
        
    Returns:
        Mesh response model
    """
    # Convert statistics to the response model
    statistics = MeshStatistics(
        num_nodes=mesh_data.statistics.get("num_nodes"),
        num_elements=mesh_data.statistics.get("num_elements"),
        element_types=mesh_data.statistics.get("element_types"),
        physical_groups=mesh_data.statistics.get("physical_groups"),
        bounding_box=mesh_data.statistics.get("bounding_box")
    )
    
    # Create the response
    return MeshResponse(
        mesh_id=mesh_data.mesh_id,
        statistics=statistics,
        output_file=str(mesh_data.output_file) if mesh_data.output_file else None,
        visualization_file=str(mesh_data.visualization_data) if mesh_data.visualization_data else None,
        success=success,
        message=message
    )


@app.post("/api/mesh/generate", response_model=MeshResponse)
async def generate_mesh(request: MeshRequest) -> MeshResponse:
    """
    Generate a mesh from a natural language prompt.
    
    Args:
        request: Mesh generation request
        
    Returns:
        Mesh generation response
    """
    try:
        # Create a mesh generator with the specified LLM config
        generator = MeshGenerator(
            llm_provider=request.llm_config.provider if request.llm_config else "ollama",
            model_name=request.llm_config.model_name if request.llm_config else None,
            temperature=request.llm_config.temperature if request.llm_config else 0.2,
            max_tokens=request.llm_config.max_tokens if request.llm_config else 2048,
            api_key=request.llm_config.api_key if request.llm_config else None,
            host=request.llm_config.host if request.llm_config else None
        )
        
        # Generate the mesh
        mesh_data = await generator.generate(request.prompt, request.system_prompt)
        
        # Store the generator for future use
        mesh_generators[mesh_data.mesh_id] = generator
        
        # Return the response
        return mesh_data_to_response(
            mesh_data,
            success=True,
            message="Mesh generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mesh/refine", response_model=MeshResponse)
async def refine_mesh(request: MeshRefinementRequest) -> MeshResponse:
    """
    Refine a mesh based on feedback.
    
    Args:
        request: Mesh refinement request
        
    Returns:
        Mesh refinement response
    """
    try:
        # Get the mesh generator
        generator = get_mesh_generator(request.mesh_id)
        
        # Load the mesh
        mesh_file = os.path.join(OUTPUT_DIR, f"{request.mesh_id}.msh")
        if not os.path.exists(mesh_file):
            raise HTTPException(status_code=404, detail=f"Mesh with ID {request.mesh_id} not found")
        
        mesh_data = generator.load_mesh(mesh_file)
        
        # Refine the mesh
        refined_mesh_data = await generator.refine(mesh_data, request.feedback)
        
        # Return the response
        return mesh_data_to_response(
            refined_mesh_data,
            success=True,
            message="Mesh refined successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the LLM about mesh generation.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response
    """
    try:
        # Get the mesh generator
        generator = get_mesh_generator(request.mesh_id)
        
        # Chat with the LLM
        response = await generator.chat(request.message)
        
        # Return the response
        return ChatResponse(
            response=response,
            mesh_id=request.mesh_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mesh/{mesh_id}", response_model=MeshResponse)
async def get_mesh(mesh_id: str) -> MeshResponse:
    """
    Get information about a mesh.
    
    Args:
        mesh_id: ID of the mesh to get
        
    Returns:
        Mesh information
    """
    try:
        # Check if the mesh exists
        mesh_file = os.path.join(OUTPUT_DIR, f"{mesh_id}.msh")
        if not os.path.exists(mesh_file):
            raise HTTPException(status_code=404, detail=f"Mesh with ID {mesh_id} not found")
        
        # Get the mesh generator
        generator = get_mesh_generator(mesh_id)
        
        # Load the mesh
        mesh_data = generator.load_mesh(mesh_file)
        
        # Return the response
        return mesh_data_to_response(
            mesh_data,
            success=True,
            message="Mesh retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mesh/{mesh_id}/download")
async def download_mesh(mesh_id: str, format: str = "msh"):
    """
    Download a mesh file.
    
    Args:
        mesh_id: ID of the mesh to download
        format: Format to download (msh, vtk, etc.)
        
    Returns:
        Mesh file
    """
    try:
        # Check if the mesh exists
        mesh_file = os.path.join(OUTPUT_DIR, f"{mesh_id}.msh")
        if not os.path.exists(mesh_file):
            raise HTTPException(status_code=404, detail=f"Mesh with ID {mesh_id} not found")
        
        # Get the mesh generator
        generator = get_mesh_generator(mesh_id)
        
        # Load the mesh
        mesh_data = generator.load_mesh(mesh_file)
        
        # Export the mesh in the requested format
        if format != "msh":
            export_file = generator.export_mesh(mesh_data, format)
        else:
            export_file = mesh_file
        
        # Return the file
        return FileResponse(
            export_file,
            media_type="application/octet-stream",
            filename=f"{mesh_id}.{format}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mesh/{mesh_id}/visualization")
async def get_visualization(mesh_id: str):
    """
    Get the visualization for a mesh.
    
    Args:
        mesh_id: ID of the mesh to visualize
        
    Returns:
        Visualization image
    """
    try:
        # Check if the visualization exists
        vis_file = os.path.join(OUTPUT_DIR, f"{mesh_id}_vis.png")
        if not os.path.exists(vis_file):
            raise HTTPException(status_code=404, detail=f"Visualization for mesh with ID {mesh_id} not found")
        
        # Return the file
        return FileResponse(
            vis_file,
            media_type="image/png",
            filename=f"{mesh_id}_vis.png"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files for the web UI
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# Serve static files for the web UI
try:
    app.mount("/", StaticFiles(directory="web", html=True), name="web")
except RuntimeError:
    # Web UI not available, just log a message
    print("Web UI not available. API endpoints will still work.") 