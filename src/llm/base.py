"""
Base LLM interface for the Gmsh MCP system.
Defines the common interface for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import re


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    # Define mesh types for examples
    MESH_EXAMPLES = {
        "cube_with_hole": {
            "name": "Cube with cylindrical hole",
            "code": """# Set mesh options
gmsh.option.setNumber('Mesh.Algorithm3D', 1)
gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
gmsh.option.setNumber('Mesh.Optimize', 1)

# Create main cube centered at origin
L = 5  # side length
box = gmsh.model.occ.addBox(-L/2, -L/2, -L/2, L, L, L)

# Create cylinder for the hole
R = 1  # radius
# Make cylinder slightly longer than cube to ensure complete cut
cyl = gmsh.model.occ.addCylinder(-L, 0, 0, 2*L, 0, 0, R)

# Boolean operation to create the hole
gmsh.model.occ.cut([(3, box)], [(3, cyl)])
gmsh.model.occ.synchronize()

# Set mesh sizes
base_size = L / 10
fine_size = base_size / 2

# Get edges on the cylindrical hole
curves = gmsh.model.getEntities(1)
for dim, tag in curves:
    # Get curve type
    curve_type = gmsh.model.getType(dim, tag)
    # Use finer mesh size for circular edges
    if curve_type == "Circle":
        gmsh.model.mesh.setSize([(dim, tag)], fine_size)
    else:
        gmsh.model.mesh.setSize([(dim, tag)], base_size)

# Generate mesh
gmsh.model.mesh.generate(3)

# Save the mesh to a file (IMPORTANT: do this BEFORE finalizing)
gmsh.write("cube_with_hole.msh")"""
        },
        "sphere": {
            "name": "Sphere with refined surface",
            "code": """# Set mesh options
gmsh.option.setNumber('Mesh.Algorithm3D', 1)
gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
gmsh.option.setNumber('Mesh.Optimize', 1)

# Create a sphere centered at origin
R = 5  # radius
sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
gmsh.model.occ.synchronize()

# Set mesh sizes
base_size = R / 8
fine_size = base_size / 2

# Get surface entities
surfaces = gmsh.model.getEntities(2)
for dim, tag in surfaces:
    gmsh.model.mesh.setSize([(dim, tag)], fine_size)

# Generate mesh
gmsh.model.mesh.generate(3)

# Save the mesh to a file
gmsh.write("sphere.msh")"""
        },
        "complex_shape": {
            "name": "Complex shape with multiple boolean operations",
            "code": """# Set mesh options
gmsh.option.setNumber('Mesh.Algorithm3D', 1)
gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
gmsh.option.setNumber('Mesh.Optimize', 1)

# Create base shapes
L = 10  # base size
box1 = gmsh.model.occ.addBox(-L/2, -L/2, -L/2, L, L, L)
box2 = gmsh.model.occ.addBox(0, -L/4, -L/4, L/2, L/2, L/2)
cyl = gmsh.model.occ.addCylinder(-L/2, 0, 0, L, 0, 0, L/4)
sphere = gmsh.model.occ.addSphere(0, 0, 0, L/3)

# Boolean operations to create complex shape
fused = gmsh.model.occ.fuse([(3, box1)], [(3, sphere)])[0]
cut1 = gmsh.model.occ.cut(fused, [(3, box2)])[0]
final = gmsh.model.occ.cut(cut1, [(3, cyl)])[0]
gmsh.model.occ.synchronize()

# Set mesh sizes based on feature size
base_size = L / 15
fine_size = base_size / 2

# Apply size field for refinement
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", [t[1] for t in gmsh.model.getEntities(1) if gmsh.model.getType(t[0], t[1]) == "Circle"])

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "IField", 1)
gmsh.model.mesh.field.setNumber(2, "LcMin", fine_size)
gmsh.model.mesh.field.setNumber(2, "LcMax", base_size)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Generate mesh
gmsh.model.mesh.generate(3)

# Save the mesh to a file
gmsh.write("complex_shape.msh")"""
        }
    }
    
    def __init__(self, model_name: str = None, temperature: float = 0.2, max_tokens: int = 2048):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: The name of the model to use
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: Optional system prompt to guide the LLM's behavior
            
        Returns:
            The generated response as a string
        """
        pass
    
    @abstractmethod
    async def generate_code(self, prompt: str, feedback: Optional[str] = None) -> str:
        """
        Generate code from the LLM, optimized for code generation.
        
        Args:
            prompt: The user prompt describing the code to generate
            feedback: Optional feedback from previous attempts
            
        Returns:
            The generated code as a string
        """
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Have a multi-turn conversation with the LLM.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response as a string
        """
        pass
    
    def extract_code_block(self, text: str) -> str:
        """
        Extract a Python code block from the LLM response.
        
        Args:
            text: The full text response from the LLM
            
        Returns:
            The extracted Python code, or the original text if no code block is found
        """
        # Look for Python code blocks (```python ... ```)
        if "```python" in text and "```" in text.split("```python", 1)[1]:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        
        # Look for generic code blocks (``` ... ```)
        if "```" in text and "```" in text.split("```", 1)[1]:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        
        return text
    
    def post_process_code(self, code: str) -> str:
        """
        Post-process generated code to clean it up.
        
        Args:
            code: The generated code
            
        Returns:
            Cleaned up code
        """
        # Extract code from markdown blocks if present
        code_match = re.search(r"```(?:python|gmsh)?\n(.*?)\n```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        
        # Remove any imports or initialization code
        code = re.sub(r'import\s+gmsh\s*\n', '', code)
        code = re.sub(r'gmsh\.initialize\(\)\s*\n', '', code)
        code = re.sub(r'gmsh\.finalize\(\)\s*\n', '', code)
        
        # Remove any comments about mesh quality settings
        code = re.sub(r'#.*mesh.*quality.*settings.*\n', '', code)
        code = re.sub(r'#.*Set.*appropriate.*mesh.*size.*\n', '', code)
        
        # Remove any instructions from the code
        code = re.sub(r'\d+\.\s+.*\n', '', code)
        code = re.sub(r'Example format:.*\n', '', code)
        
        # Remove the instructions from the prompt
        code = re.sub(r'The code should:.*?Example for a cube:', '', code, flags=re.DOTALL)
        
        return code.strip()
    
    def get_mesh_example(self, mesh_type: str = "cube_with_hole") -> str:
        """
        Get an example for a specific mesh type.
        
        Args:
            mesh_type: The type of mesh example to get
            
        Returns:
            Example code for the specified mesh type
        """
        if mesh_type in self.MESH_EXAMPLES:
            return self.MESH_EXAMPLES[mesh_type]["code"]
        return self.MESH_EXAMPLES["cube_with_hole"]["code"]
    
    def get_mesh_example_by_prompt(self, prompt: str) -> Tuple[str, str]:
        """
        Select an appropriate mesh example based on the prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Tuple of (example_name, example_code)
        """
        # Simple keyword matching to select an appropriate example
        prompt_lower = prompt.lower()
        
        if "sphere" in prompt_lower:
            example_type = "sphere"
        elif "complex" in prompt_lower or "multiple" in prompt_lower or "boolean" in prompt_lower:
            example_type = "complex_shape"
        else:
            # Default to cube with hole
            example_type = "cube_with_hole"
        
        example = self.MESH_EXAMPLES[example_type]
        return example["name"], example["code"]
    
    def get_system_prompt(self) -> str:
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
    
    def get_mesh_prompt_template(self, prompt: str, feedback: Optional[str] = None) -> str:
        """
        Get the prompt template for mesh generation.
        
        Args:
            prompt: The user prompt
            feedback: Optional feedback from previous attempts
            
        Returns:
            Formatted prompt template
        """
        # Select an appropriate example based on the prompt
        example_name, example_code = self.get_mesh_example_by_prompt(prompt)
        
        # Add feedback if provided
        feedback_section = ""
        if feedback:
            feedback_section = f"""
Previous attempt had the following issues:
{feedback}

Please address these issues in your new code.
"""
        
        return f"""
Generate Gmsh Python API code to create a mesh based on this description: {prompt}
{feedback_section}
The code should follow these requirements:

Mesh Settings:
1. Set mesh algorithm to Netgen (Mesh.Algorithm3D = 1)
2. Enable mesh optimization (Mesh.OptimizeNetgen = 1, Mesh.Optimize = 1)
3. Set appropriate mesh size based on model dimensions (typically 1/10th of smallest dimension)
4. Add refinement around critical features:
   - Use smaller mesh size near curved surfaces
   - Add refinement near intersections and edges
   - Ensure smooth transition between different mesh sizes

Geometric Requirements:
1. For boolean operations (e.g. holes, cuts):
   - Center holes/cuts relative to the main shape unless specified otherwise
   - Ensure cuts go completely through the shape
   - Use appropriate boolean operations (cut, fuse, intersect)
   - IMPORTANT: Boolean operations use tuples of (dim, tag), NOT lists. Example: [(3, box)], [(3, cyl)]
2. For positioning:
   - Default to centering shapes at origin (0,0,0) unless specified otherwise
   - Maintain symmetry in operations unless asymmetry is explicitly requested
3. For dimensions:
   - Use exact dimensions as specified in the prompt
   - For cylindrical features, ensure radius and length are correct
   - For rectangular features, ensure width, height, and depth are correct

Example for a {example_name}:
```python
{example_code}
```

Generate code that follows these guidelines and requirements.

Use the following Gmsh Python API functions:
- For creating a box: gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
- For creating a cylinder: gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r)
- For creating a sphere: gmsh.model.occ.addSphere(x, y, z, r)
- For boolean operations:
  * gmsh.model.occ.cut([(3, target)], [(3, tool)])
  * gmsh.model.occ.fuse([(3, obj1)], [(3, obj2)])
  * gmsh.model.occ.intersect([(3, obj1)], [(3, obj2)])
- Always call gmsh.model.occ.synchronize() after geometry operations
- For mesh generation: gmsh.model.mesh.generate(3)
- For setting mesh size: gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
""" 