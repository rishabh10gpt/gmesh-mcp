"""
Base class for LLM implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import re
import textwrap

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self):
        """Initialize the base LLM."""
        self.system_prompt = """You are an expert in Gmsh, a three-dimensional finite element mesh generator.
Your task is to generate Gmsh Python code based on natural language prompts.

The code should:
1. Initialize Gmsh
2. Create the requested geometry with accurate dimensions
3. Generate a high-quality tetrahedral mesh
4. Write the mesh to a file
5. Finalize Gmsh

IMPORTANT GUIDELINES:
- Always use gmsh.model.occ for geometry creation (OpenCASCADE kernel)
- Set appropriate mesh size parameters for good quality
- Use boolean operations (cut, fuse, etc.) for complex geometries
- Always synchronize the model after geometry operations
- Include mesh optimization steps
- Handle errors gracefully
- Ensure the mesh is saved with gmsh.write() before finalizing
- Make sure to properly close Gmsh with gmsh.finalize()

For mesh quality:
- Use gmsh.option.setNumber("Mesh.Algorithm3D", 1) for Delaunay algorithm
- Set gmsh.option.setNumber("Mesh.OptimizeNetgen", 1) for Netgen optimizer
- Use appropriate mesh size controls
- Ensure smooth transitions between different mesh regions
- Refine mesh near curved surfaces and complex features

Example of good code structure:
```python
import gmsh

# Initialize Gmsh
gmsh.initialize()

# Set mesh quality options
gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay for volumes
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Netgen for 3D
gmsh.option.setNumber("Mesh.Optimize", 1)  # Optimize mesh
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Optimize with Netgen
gmsh.option.setNumber("Mesh.QualityType", 2)  # SICN quality measure
gmsh.option.setNumber("Mesh.MinimumCirclePoints", 6)  # Minimum points on circles
gmsh.option.setNumber("Mesh.MinimumCurvePoints", 6)  # Minimum points on curves
gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 6)  # Minimum elements per 2π

# Create geometry using OpenCASCADE kernel
cube = gmsh.model.occ.addBox(-5, -5, -5, 10, 10, 10)
cylinder = gmsh.model.occ.addCylinder(0, 0, -6, 0, 0, 12, 1)
gmsh.model.occ.cut([(3, cube)], [(3, cylinder)])
gmsh.model.occ.synchronize()

# Generate mesh
gmsh.model.mesh.generate(3)

# Write mesh to file
gmsh.write("output.msh")

# Finalize Gmsh
gmsh.finalize()
```"""
    
    @abstractmethod
    async def generate_code(self, prompt: str, feedback: Optional[str] = None) -> str:
        """Generate Gmsh code from a prompt."""
        pass
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Find code blocks
        code_start = response.find("```python")
        if code_start == -1:
            code_start = response.find("```")
        
        if code_start != -1:
            code_end = response.find("```", code_start + 3)
            if code_end != -1:
                code = response[code_start:code_end].strip()
                code = code.replace("```python", "").replace("```", "").strip()
                return code
        
        # If no code blocks, try to extract Python code
        python_code = re.search(r'import\s+gmsh.*?gmsh\.finalize\(\)', response, re.DOTALL)
        if python_code:
            return python_code.group(0)
        
        return response.strip()
    
    def _add_mesh_quality_settings(self, code: str) -> str:
        """Add mesh quality settings to the generated code.
        
        Args:
            code (str): The generated Gmsh code
            
        Returns:
            str: The code with mesh quality settings added
        """
        # Remove any existing imports and quality settings
        code = re.sub(r'import gmsh\n+', '', code)
        code = re.sub(r'gmsh\.finalize\(\)', '', code)
        
        # Clean up code
        code = textwrap.dedent(code)
        
        # Split into lines and remove empty lines
        lines = [line for line in code.split('\n') if line.strip()]
        
        # Indent all lines
        indented_lines = ['    ' + line for line in lines]
        
        # Create the final structure with proper escaping
        template = '''import gmsh

try:
    # Initialize Gmsh
    gmsh.initialize()
    
    # Set mesh quality options
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay for volumes
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Netgen for 3D
    gmsh.option.setNumber("Mesh.Optimize", 1)  # Optimize mesh
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Optimize with Netgen
    gmsh.option.setNumber("Mesh.QualityType", 2)  # SICN quality measure
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)  # More points on circles
    gmsh.option.setNumber("Mesh.MinimumCurvePoints", 12)  # More points on curves
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)  # More elements per 2π
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Use second-order elements
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # Curved second-order elements
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)  # Optimize high-order elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)  # Minimum element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0)  # Maximum element size
    
{0}

    # Generate mesh
    gmsh.model.mesh.generate(3)
    
    # Write mesh to file
    gmsh.write("output.msh")

except Exception as e:
    print("Error: " + str(e))
finally:
    gmsh.finalize()'''

        # Format the template with the indented code
        return template.format('\n'.join(indented_lines)) 