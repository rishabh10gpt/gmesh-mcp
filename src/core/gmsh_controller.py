"""
Gmsh controller for the MCP system.
Handles interaction with the Gmsh API for mesh generation.
"""

import os
import sys
import tempfile
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
import json
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Import Gmsh if available
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

from ..utils.config import OUTPUT_DIR, GMSH_EXECUTABLE_PATH, VISUALIZATION_ENABLED


class MeshData:
    """Class to store mesh data and provide utility methods."""
    
    def __init__(self, mesh_id: str = None):
        """
        Initialize mesh data.
        
        Args:
            mesh_id: Unique identifier for the mesh (generated if not provided)
        """
        self.mesh_id = mesh_id or str(uuid.uuid4())
        self.nodes = []  # List of node coordinates
        self.elements = []  # List of element connectivity
        self.element_types = []  # List of element types
        self.boundaries = []  # List of boundary elements
        self.physical_groups = {}  # Dictionary of physical groups
        self.statistics = {}  # Mesh statistics
        self.output_file = None  # Path to output file
        self.visualization_data = None  # Visualization data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert mesh data to a dictionary.
        
        Returns:
            Dictionary representation of the mesh data
        """
        return {
            "mesh_id": self.mesh_id,
            "statistics": self.statistics,
            "output_file": str(self.output_file) if self.output_file else None,
            "physical_groups": self.physical_groups
        }
    
    def from_dict(self, data: Dict[str, Any]) -> 'MeshData':
        """
        Load mesh data from a dictionary.
        
        Args:
            data: Dictionary representation of the mesh data
            
        Returns:
            Self for method chaining
        """
        self.mesh_id = data.get("mesh_id", self.mesh_id)
        self.statistics = data.get("statistics", {})
        self.output_file = Path(data["output_file"]) if data.get("output_file") else None
        self.physical_groups = data.get("physical_groups", {})
        return self
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save mesh data to a file.
        
        Args:
            filepath: Path to save the mesh data (generated if not provided)
            
        Returns:
            Path to the saved file
        """
        if not filepath:
            filepath = os.path.join(OUTPUT_DIR, f"{self.mesh_id}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'MeshData':
        """
        Load mesh data from a file.
        
        Args:
            filepath: Path to the mesh data file
            
        Returns:
            Loaded mesh data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls().from_dict(data)


class GmshController:
    """Controller for Gmsh operations."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """
        Initialize the Gmsh controller.
        
        Args:
            executable_path: Path to the Gmsh executable (defaults to environment variable)
        """
        self.executable_path = executable_path or GMSH_EXECUTABLE_PATH
        self.initialized = False
        self.current_mesh = None
    
    def initialize(self) -> bool:
        """
        Initialize Gmsh.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True
        
        if not GMSH_AVAILABLE:
            raise ImportError("Gmsh is not available. Please install it with 'pip install gmsh'.")
        
        try:
            gmsh.initialize()
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Gmsh: {e}")
            return False
    
    def finalize(self) -> None:
        """Finalize Gmsh."""
        if self.initialized:
            try:
                gmsh.finalize()
                self.initialized = False
            except Exception as e:
                print(f"Error finalizing Gmsh: {e}")
    
    def __enter__(self) -> 'GmshController':
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finalize()
    
    def execute_code(self, code: str) -> Tuple[bool, str, MeshData]:
        """
        Execute Gmsh Python code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (success, output, mesh_data)
        """
        # Create a new mesh data object
        mesh_data = MeshData()
        
        # Capture stdout and stderr
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        # Initialize Gmsh if not already initialized
        was_initialized = self.initialized
        if not was_initialized:
            self.initialize()
        
        success = True
        try:
            # Execute the code in a safe environment
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Add mesh_data to the local variables
                local_vars = {"gmsh": gmsh, "mesh_data": mesh_data}
                
                # Execute the code
                exec(code, globals(), local_vars)
                
                # Collect mesh statistics
                mesh_data.statistics = self._collect_statistics()
                
                # Save the mesh to a file
                output_file = os.path.join(OUTPUT_DIR, f"{mesh_data.mesh_id}.msh")
                gmsh.write(output_file)
                mesh_data.output_file = output_file
                
                # Generate visualization if enabled
                if VISUALIZATION_ENABLED:
                    self._generate_visualization(mesh_data)
        except Exception as e:
            success = False
            with redirect_stderr(stderr_buffer):
                print(f"Error executing Gmsh code: {e}")
        finally:
            # Finalize Gmsh if it wasn't initialized before
            if not was_initialized:
                self.finalize()
        
        # Get the output
        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()
        output = stdout + "\n" + stderr if stderr else stdout
        
        return success, output, mesh_data
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """
        Collect mesh statistics from Gmsh.
        
        Returns:
            Dictionary of mesh statistics
        """
        stats = {}
        
        try:
            # Get the number of nodes and elements
            stats["num_nodes"] = gmsh.model.mesh.getNodeCount()
            stats["num_elements"] = gmsh.model.mesh.getElementCount()
            
            # Get the element types
            element_types = set()
            for entity in gmsh.model.getEntities():
                dim, tag = entity
                element_types.update(gmsh.model.mesh.getElementTypes(dim, tag))
            
            # Convert element types to names
            element_type_names = []
            for element_type in element_types:
                element_name = gmsh.model.mesh.getElementProperties(element_type)[0]
                element_type_names.append(element_name)
            
            stats["element_types"] = element_type_names
            
            # Get the physical groups
            physical_groups = {}
            for dim in range(4):  # 0D to 3D
                for tag in gmsh.model.getPhysicalGroups(dim):
                    name = gmsh.model.getPhysicalName(dim, tag[1])
                    if not name:
                        name = f"Physical Group {tag[1]}"
                    physical_groups[name] = {"dimension": dim, "tag": tag[1]}
            
            stats["physical_groups"] = physical_groups
            
            # Get the bounding box
            bounding_box = []
            for entity in gmsh.model.getEntities():
                dim, tag = entity
                bbox = gmsh.model.getBoundingBox(dim, tag)
                if bbox:
                    bounding_box.append(bbox)
            
            if bounding_box:
                # Combine bounding boxes
                min_x = min(bbox[0] for bbox in bounding_box)
                min_y = min(bbox[1] for bbox in bounding_box)
                min_z = min(bbox[2] for bbox in bounding_box)
                max_x = max(bbox[3] for bbox in bounding_box)
                max_y = max(bbox[4] for bbox in bounding_box)
                max_z = max(bbox[5] for bbox in bounding_box)
                
                stats["bounding_box"] = {
                    "min": [min_x, min_y, min_z],
                    "max": [max_x, max_y, max_z]
                }
        except Exception as e:
            print(f"Error collecting mesh statistics: {e}")
        
        return stats
    
    def _generate_visualization(self, mesh_data: MeshData) -> None:
        """
        Generate visualization data for the mesh.
        
        Args:
            mesh_data: Mesh data to visualize
        """
        try:
            # Create a visualization file
            vis_file = os.path.join(OUTPUT_DIR, f"{mesh_data.mesh_id}_vis.png")
            
            # Use Gmsh's built-in visualization capabilities
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.GraphicsWidth", 800)
            gmsh.option.setNumber("General.GraphicsHeight", 600)
            
            # Set visualization options
            gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 1)
            gmsh.option.setNumber("Mesh.ColorCarousel", 2)  # Random colors
            
            # Create the image
            gmsh.write(vis_file.replace(".png", ".geo_unrolled"))
            gmsh.fltk.initialize()
            gmsh.fltk.update()
            gmsh.fltk.screenshot(vis_file)
            gmsh.fltk.finalize()
            
            # Store the visualization file path
            mesh_data.visualization_data = vis_file
        except Exception as e:
            print(f"Error generating visualization: {e}")
    
    def load_mesh(self, filepath: str) -> MeshData:
        """
        Load a mesh from a file.
        
        Args:
            filepath: Path to the mesh file
            
        Returns:
            Loaded mesh data
        """
        # Initialize Gmsh if not already initialized
        was_initialized = self.initialized
        if not was_initialized:
            self.initialize()
        
        try:
            # Clear any existing model
            gmsh.model.remove()
            
            # Load the mesh
            gmsh.open(filepath)
            
            # Create a new mesh data object
            mesh_data = MeshData()
            mesh_data.output_file = filepath
            
            # Collect mesh statistics
            mesh_data.statistics = self._collect_statistics()
            
            # Generate visualization if enabled
            if VISUALIZATION_ENABLED:
                self._generate_visualization(mesh_data)
            
            return mesh_data
        finally:
            # Finalize Gmsh if it wasn't initialized before
            if not was_initialized:
                self.finalize()
    
    def export_mesh(self, mesh_data: MeshData, format: str = "msh") -> str:
        """
        Export a mesh to a file in the specified format.
        
        Args:
            mesh_data: Mesh data to export
            format: Export format (msh, vtk, etc.)
            
        Returns:
            Path to the exported file
        """
        if not mesh_data.output_file:
            raise ValueError("No mesh file to export")
        
        # Initialize Gmsh if not already initialized
        was_initialized = self.initialized
        if not was_initialized:
            self.initialize()
        
        try:
            # Load the mesh
            gmsh.open(str(mesh_data.output_file))
            
            # Export the mesh
            export_file = os.path.join(OUTPUT_DIR, f"{mesh_data.mesh_id}.{format}")
            gmsh.write(export_file)
            
            return export_file
        finally:
            # Finalize Gmsh if it wasn't initialized before
            if not was_initialized:
                self.finalize() 