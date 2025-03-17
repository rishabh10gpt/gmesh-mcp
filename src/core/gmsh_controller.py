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
import re
import shutil
import traceback
import threading

# Import Gmsh if available
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

from ..utils.config import OUTPUT_DIR, GMSH_EXECUTABLE_PATH, VISUALIZATION_ENABLED


class MeshData:
    """Class to store mesh data and provide utility methods."""
    
    def __init__(self, mesh_id: str = None, output_file: str = None, statistics: Dict[str, Any] = None):
        """
        Initialize mesh data.
        
        Args:
            mesh_id: Unique identifier for the mesh (generated if not provided)
            output_file: Path to the output mesh file
            statistics: Dictionary of mesh statistics
        """
        self.mesh_id = mesh_id or str(uuid.uuid4())
        self.nodes = []  # List of node coordinates
        self.elements = []  # List of element connectivity
        self.element_types = []  # List of element types
        self.boundaries = []  # List of boundary elements
        self.physical_groups = {}  # Dictionary of physical groups
        self.statistics = statistics or {}  # Mesh statistics
        self.output_file = output_file  # Path to output file
        self.visualization_data = None  # Visualization data
        self.output = ""  # Output from the mesh generation process
    
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
        self.output_dir = OUTPUT_DIR
        self.meshes_dir = os.path.join(self.output_dir, "meshes")
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        self.conversations_dir = os.path.join(self.output_dir, "conversations")
        
        # Create necessary directories
        for directory in [self.output_dir, self.meshes_dir, self.visualizations_dir, self.conversations_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Store Gmsh module but don't initialize yet
        try:
            import gmsh
            self.gmsh = gmsh
        except ImportError:
            print("Warning: Gmsh Python API not found. Some functionality may be limited.")
            self.gmsh = None
    
    def initialize(self) -> bool:
        """
        Initialize Gmsh if not already initialized.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True
        
        if not GMSH_AVAILABLE:
            raise ImportError("Gmsh is not available. Please install it with 'pip install gmsh'.")
        
        try:
            # Only initialize if we're in the main thread
            if threading.current_thread() is threading.main_thread():
                self.gmsh.initialize()
                self.initialized = True
            return True
        except Exception as e:
            print(f"Warning: Could not initialize Gmsh: {e}")
            return False
    
    def finalize(self) -> None:
        """Finalize Gmsh if it was initialized."""
        if self.initialized and threading.current_thread() is threading.main_thread():
            try:
                self.gmsh.finalize()
            except Exception as e:
                print(f"Warning: Could not finalize Gmsh: {e}")
            finally:
                self.initialized = False
    
    def __enter__(self) -> 'GmshController':
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finalize()
    
    def execute_code(self, code: str, mesh_id: str = None) -> Tuple[bool, str]:
        """Execute the Gmsh code and return the output."""
        if mesh_id is None:
            mesh_id = str(uuid.uuid4())
            
        # Create a temporary file for the Gmsh code
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            # Ensure output path exists
            output_path = os.path.join(self.meshes_dir, f"{mesh_id}.msh")
            
            # Replace any hardcoded output path with our dynamic one
            if 'gmsh.write(' in code:
                code = re.sub(r'gmsh.write\(.*?\)', f'gmsh.write("{output_path}")', code)
            
            # Remove any existing initialization and finalization code
            code = re.sub(r'import\s+gmsh\s*\n', '', code)
            code = re.sub(r'gmsh\.initialize\(\)\s*\n', '', code)
            code = re.sub(r'gmsh\.finalize\(\)\s*\n', '', code)
            
            # Add initialization code at the beginning
            init_code = """
import gmsh
try:
    gmsh.initialize()
"""
            # Add finalization code at the end with explicit write statement
            finalize_code = """
    # Ensure the mesh is written to file
    gmsh.write("{}")
finally:
    try:
        gmsh.finalize()
    except:
        pass
""".format(output_path)
            
            # Indent the code to match the try block
            indented_code = ""
            for line in code.split('\n'):
                indented_code += "    " + line + "\n"
            
            # Write the complete code to the temporary file
            f.write(init_code + indented_code + finalize_code)
            f.flush()
            
            # Execute the code in a subprocess to avoid Gmsh initialization issues
            try:
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ)
                )
                
                output = result.stdout + result.stderr
                success = result.returncode == 0 and os.path.exists(output_path)
                
                return success, output
                
            except Exception as e:
                return False, f"Error executing Gmsh code: {str(e)}\n{traceback.format_exc()}"
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(f.name)
                except:
                    pass
    
    def _collect_statistics(self, mesh_data: MeshData) -> None:
        """
        Collect mesh statistics from Gmsh.
        
        Args:
            mesh_data: Mesh data to collect statistics from
        """
        # First try to extract statistics from stdout
        stdout_buffer = StringIO()
        with redirect_stdout(stdout_buffer):
            try:
                gmsh.model.mesh.renumberNodes()
                gmsh.model.mesh.renumberElements()
            except Exception as e:
                print(f"Warning: Could not renumber mesh: {e}")
        
        stdout = stdout_buffer.getvalue()
        
        # Parse node and element counts from stdout
        node_element_match = re.search(r"Info\s*:\s*(\d+)\s+nodes\s+(\d+)\s+elements", stdout)
        if node_element_match:
            mesh_data.statistics["num_nodes"] = int(node_element_match.group(1))
            mesh_data.statistics["num_elements"] = int(node_element_match.group(2))
            print(f"Extracted from stdout: {mesh_data.statistics['num_nodes']} nodes, {mesh_data.statistics['num_elements']} elements")
        
        # Parse element quality statistics from stdout
        quality_stats = {}
        quality_pattern = r"Info\s*:\s*([\d.]+)\s*<\s*quality\s*<\s*([\d.]+)\s*:\s*(\d+)\s*elements"
        for match in re.finditer(quality_pattern, stdout):
            min_qual, max_qual, count = match.groups()
            range_key = f"{min_qual}-{max_qual}"
            quality_stats[range_key] = int(count)
        
        if quality_stats:
            mesh_data.statistics["quality_distribution"] = quality_stats
            
            # Calculate overall quality metrics
            total_elements = sum(quality_stats.values())
            weighted_quality = 0
            for range_str, count in quality_stats.items():
                min_qual, max_qual = map(float, range_str.split('-'))
                avg_qual = (min_qual + max_qual) / 2
                weighted_quality += avg_qual * count
            
            mesh_data.statistics["average_quality"] = weighted_quality / total_elements if total_elements > 0 else 0
        
        # Parse worst element quality
        worst_match = re.search(r"worst\s*=\s*([\d.]+)", stdout)
        if worst_match:
            mesh_data.statistics["worst_element_quality"] = float(worst_match.group(1))
        
        # Parse optimization results for quality metrics
        opt_match = re.search(r"Optimization starts.*?with worst = ([\d.]+) / average = ([\d.]+)", stdout, re.DOTALL)
        if opt_match:
            if not mesh_data.statistics.get("worst_element_quality"):
                mesh_data.statistics["worst_element_quality"] = float(opt_match.group(1))
                print(f"Extracted worst element quality from optimization: {mesh_data.statistics['worst_element_quality']}")
            if not mesh_data.statistics.get("average_quality"):
                mesh_data.statistics["average_quality"] = float(opt_match.group(2))
                print(f"Extracted average quality from optimization: {mesh_data.statistics['average_quality']}")
        
        # If we couldn't get statistics from stdout, try using the API
        if not mesh_data.statistics.get("num_nodes") or not mesh_data.statistics.get("num_elements"):
            try:
                # Get number of nodes and elements
                nodes = gmsh.model.mesh.getNodes()
                elements = gmsh.model.mesh.getElements()
                
                if nodes and nodes[0].size > 0:
                    mesh_data.statistics["num_nodes"] = nodes[0].shape[0]
                
                if elements and elements[0].size > 0:
                    mesh_data.statistics["num_elements"] = sum(tag.shape[0] for tag in elements[1])
                
                # Get element types
                element_types = gmsh.model.mesh.getElementTypes()
                mesh_data.statistics["element_types"] = [gmsh.model.mesh.getElementTypeName(t) for t in element_types]
                
                # Get element quality statistics if not already extracted
                if not mesh_data.statistics.get("quality_distribution"):
                    quality_stats = {}
                    all_qualities = []
                    
                    for element_type in element_types:
                        elements = gmsh.model.mesh.getElementsByType(element_type)[0]
                        if elements.size > 0:
                            qualities = gmsh.model.mesh.getElementQualities(elements)
                            if qualities.size > 0:
                                all_qualities.append(qualities)
                                
                                # Calculate quality ranges
                                ranges = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), 
                                        (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                                for min_qual, max_qual in ranges:
                                    count = np.sum((qualities >= min_qual) & (qualities < max_qual))
                                    if count > 0:
                                        range_key = f"{min_qual}-{max_qual}"
                                        quality_stats[range_key] = quality_stats.get(range_key, 0) + int(count)
                    
                    if quality_stats:
                        mesh_data.statistics["quality_distribution"] = quality_stats
                    
                    if all_qualities:
                        all_qualities = np.concatenate(all_qualities)
                        if not mesh_data.statistics.get("worst_element_quality"):
                            mesh_data.statistics["worst_element_quality"] = float(np.min(all_qualities))
                        if not mesh_data.statistics.get("average_quality"):
                            mesh_data.statistics["average_quality"] = float(np.mean(all_qualities))
                
                # Get bounding box
                try:
                    entities = gmsh.model.getEntities(3)  # Get 3D entities
                    if entities:
                        bounding_box = gmsh.model.getBoundingBox(entities[0][0], entities[0][1])
                        if bounding_box:
                            mesh_data.statistics["bounding_box"] = {
                                "min": bounding_box[:3],
                                "max": bounding_box[3:]
                            }
                except Exception as e:
                    print(f"Warning: Could not get bounding box: {e}")
            
            except Exception as e:
                print(f"Error collecting mesh statistics via API: {e}")
        
        # Set default element types if not found
        if not mesh_data.statistics.get("element_types"):
            mesh_data.statistics["element_types"] = ["Tetrahedron"]
    
    def _extract_statistics_from_output(self, mesh_data: MeshData) -> None:
        """
        Extract statistics from the output and populate the mesh_data object.
        
        Args:
            mesh_data: Mesh data to extract statistics from
        """
        # Try to extract statistics from the output
        node_element_match = re.search(r"Info\s*:\s*(\d+)\s+nodes\s+(\d+)\s+elements", mesh_data.output)
        if node_element_match:
            mesh_data.statistics["num_nodes"] = int(node_element_match.group(1))
            mesh_data.statistics["num_elements"] = int(node_element_match.group(2))
            print(f"Extracted from output: {mesh_data.statistics['num_nodes']} nodes, {mesh_data.statistics['num_elements']} elements")
        
        opt_match = re.search(r"Optimization starts.*?with worst = ([\d.]+) / average = ([\d.]+)", mesh_data.output, re.DOTALL)
        if opt_match:
            mesh_data.statistics["worst_element_quality"] = float(opt_match.group(1))
            mesh_data.statistics["average_quality"] = float(opt_match.group(2))
            print(f"Extracted quality metrics from output: worst={mesh_data.statistics['worst_element_quality']}, avg={mesh_data.statistics['average_quality']}")
        
        # Extract quality distribution
        quality_stats = {}
        quality_pattern = r"Info\s*:\s*([\d.]+)\s*<\s*quality\s*<\s*([\d.]+)\s*:\s*(\d+)\s*elements"
        for match in re.finditer(quality_pattern, mesh_data.output):
            min_qual, max_qual, count = match.groups()
            range_key = f"{min_qual}-{max_qual}"
            quality_stats[range_key] = int(count)
        
        if quality_stats:
            mesh_data.statistics["quality_distribution"] = quality_stats
            print(f"Extracted quality distribution with {len(quality_stats)} ranges")
    
    def _generate_visualization(self, mesh_data: MeshData) -> None:
        """
        Generate visualization data for the mesh.
        
        Args:
            mesh_data: Mesh data to visualize
        """
        try:
            # Create a visualization file in the visualizations directory
            vis_file = os.path.join(self.visualizations_dir, f"{mesh_data.mesh_id}_vis.png")
            
            # Create a temporary file for visualization code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
                vis_code = f"""
import gmsh
import os
import matplotlib.pyplot as plt
import numpy as np

# Initialize Gmsh
gmsh.initialize()

# Load the mesh
gmsh.open("{mesh_data.output_file}")

# Set visualization options
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.GraphicsWidth", 800)
gmsh.option.setNumber("General.GraphicsHeight", 600)
gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.VolumeEdges", 1)
gmsh.option.setNumber("Mesh.ColorCarousel", 2)

# Get mesh data for visualization
nodes = gmsh.model.mesh.getNodes()
elements = gmsh.model.mesh.getElements()

if nodes and elements:
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    if nodes[0].size > 0:
        x = nodes[0][:, 0]
        y = nodes[0][:, 1]
        z = nodes[0][:, 2]
        ax.scatter(x, y, z, c='b', marker='o', s=1)
    
    # Plot elements
    for element_type in elements[0]:
        element_nodes = gmsh.model.mesh.getElementNodes(element_type)
        if element_nodes.size > 0:
            # Reshape node coordinates for elements
            element_coords = element_nodes.reshape(-1, 3)
            x = element_coords[:, 0]
            y = element_coords[:, 1]
            z = element_coords[:, 2]
            ax.plot_trisurf(x, y, z)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Visualization')
    
    # Save the plot
    plt.savefig("{vis_file}")
    plt.close()

# Finalize Gmsh
gmsh.finalize()
"""
                f.write(vis_code)
                f.flush()
                
                # Execute visualization in a subprocess
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ)
                )
                
                if result.returncode == 0 and os.path.exists(vis_file):
                    mesh_data.visualization_data = vis_file
                else:
                    print(f"Error generating visualization: {result.stderr}")
                
        except Exception as e:
            print(f"Error generating visualization: {e}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(f.name)
            except:
                pass
    
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
            self._collect_statistics(mesh_data)
            
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

    def _validate_geometry(self, mesh_data: MeshData) -> List[str]:
        """
        Validate geometric features of the mesh.
        
        Args:
            mesh_data: Mesh data to validate
            
        Returns:
            List of geometric issues found
        """
        issues = []
        
        try:
            # Get model bounds
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            
            # Check if model is centered at origin (within tolerance)
            tolerance = 1e-3
            center_x = (xmax + xmin) / 2
            center_y = (ymax + ymin) / 2
            center_z = (zmax + zmin) / 2
            
            if abs(center_x) > tolerance or abs(center_y) > tolerance or abs(center_z) > tolerance:
                issues.append(f"Model is not centered at origin. Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
            
            # Check dimensions
            width = xmax - xmin
            height = ymax - ymin
            depth = zmax - zmin
            
            # Store dimensions for reference
            mesh_data.statistics["dimensions"] = {
                "width": width,
                "height": height,
                "depth": depth,
                "center": (center_x, center_y, center_z)
            }
            
            # Check for symmetry (if applicable)
            entities = gmsh.model.getEntities(2)  # Get all 2D entities (surfaces)
            if len(entities) > 0:
                # Check if surfaces are symmetric about main axes
                surface_centers = []
                for dim, tag in entities:
                    com = gmsh.model.occ.getCenterOfMass(dim, tag)
                    surface_centers.append(com)
                
                # Check if centers are symmetric about main planes
                for axis in range(3):  # x, y, z
                    coords = [c[axis] for c in surface_centers]
                    mean = sum(coords) / len(coords)
                    if abs(mean) > tolerance:
                        issues.append(f"Surfaces are not symmetric about {'xyz'[axis]}-axis")
            
            # Check for holes/cuts (if applicable)
            volumes = gmsh.model.getEntities(3)  # Get all 3D entities
            if len(volumes) > 1:
                # Multiple volumes might indicate incomplete boolean operations
                issues.append(f"Found {len(volumes)} separate volumes - boolean operations may not be complete")
            
            # Check for disconnected regions
            try:
                # Get physical groups
                physical_groups = gmsh.model.getPhysicalGroups()
                if len(physical_groups) > 1:
                    issues.append(f"Found {len(physical_groups)} separate physical regions")
            except:
                pass  # Physical groups might not be defined
            
        except Exception as e:
            issues.append(f"Error validating geometry: {str(e)}")
        
        return issues

    def _validate_mesh(self, mesh_data: MeshData) -> List[str]:
        """
        Validate a mesh and provide feedback.
        
        Args:
            mesh_data: Mesh data to validate
            
        Returns:
            List of quality issues, empty if mesh is good
        """
        issues = []
        
        # Check if mesh file exists
        if not mesh_data.output_file or not os.path.exists(mesh_data.output_file):
            issues.append("Mesh file was not generated successfully")
            return issues
        
        # Validate geometric features
        geometric_issues = self._validate_geometry(mesh_data)
        issues.extend(geometric_issues)
        
        # Check if we have basic mesh statistics
        if not mesh_data.statistics.get("num_nodes") or not mesh_data.statistics.get("num_elements"):
            issues.append("Failed to extract basic mesh statistics")
            return issues
        
        # Check number of elements
        num_elements = mesh_data.statistics.get("num_elements", 0)
        if num_elements < 10:
            issues.append(f"Mesh has too few elements ({num_elements})")
        
        # Check element quality
        worst_quality = mesh_data.statistics.get("worst_element_quality", 0)
        avg_quality = mesh_data.statistics.get("average_quality", 0)
        
        if worst_quality < 0.1:
            issues.append(f"Some elements have very poor quality (minimum quality: {worst_quality:.3f})")
        if avg_quality < 0.5:
            issues.append(f"Average element quality is low ({avg_quality:.3f})")
        
        # Check quality distribution
        quality_dist = mesh_data.statistics.get("quality_distribution", {})
        if quality_dist:
            poor_elements = sum(count for range_str, count in quality_dist.items() 
                              if float(range_str.split('-')[1]) < 0.3)
            if poor_elements > 0:
                issues.append(f"There are {poor_elements} elements with quality below 0.3")
        
        return issues 