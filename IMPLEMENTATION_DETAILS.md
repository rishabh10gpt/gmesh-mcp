# Gmsh MCP: Implementation Details

This document provides a comprehensive explanation of the implementation logic and working principles of the Gmsh Model Context Protocol (MCP) system.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [LLM Integration](#llm-integration)
4. [Communication Flow](#communication-flow)
5. [Mesh Generation Process](#mesh-generation-process)
6. [Iterative Refinement](#iterative-refinement)
7. [Server Implementation](#server-implementation)
8. [CLI Implementation](#cli-implementation)
9. [Data Models](#data-models)
10. [Visualization and Export](#visualization-and-export)
11. [Error Handling](#error-handling)
12. [Extension Points](#extension-points)

## System Architecture

The Gmsh MCP system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────▶│  MeshGenerator  │────▶│  LLM Provider   │
│  (CLI/API/Web)  │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │ GmshController  │
                        │                 │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │  Gmsh Library   │
                        │                 │
                        └─────────────────┘
```

This architecture enables:
- Multiple user interfaces (CLI, API, web) to interact with the same core functionality
- Support for different LLM providers (OpenAI, Anthropic, Ollama)
- Clean separation between mesh generation logic and LLM interaction
- Consistent handling of mesh data and visualization

## Core Components

### MeshGenerator

The `MeshGenerator` class is the central component that orchestrates the entire process:

```python
class MeshGenerator:
    def __init__(self, llm_provider="openai", model_name=None, temperature=0.2, **kwargs):
        self.llm = LLMFactory.create_llm(provider=llm_provider, model_name=model_name, 
                                         temperature=temperature, **kwargs)
        self.gmsh_controller = GmshController()
        self.conversation_history = []
    
    async def generate(self, prompt, system_prompt=None):
        # Generate code from prompt using LLM
        # Execute code using GmshController
        # Return mesh data
        
    async def refine(self, mesh_data, feedback):
        # Generate refinement code based on feedback and mesh statistics
        # Execute refinement code
        # Return refined mesh data
        
    async def chat(self, message):
        # Have a conversation with the LLM about mesh generation
```

Key responsibilities:
- Creating and managing the LLM instance
- Maintaining conversation history for context
- Coordinating between user input and mesh generation
- Handling the refinement process

### GmshController

The `GmshController` class handles all interactions with the Gmsh library:

```python
class GmshController:
    def __init__(self, executable_path=None):
        self.executable_path = executable_path or GMSH_EXECUTABLE_PATH
        self.initialized = False
        self.current_mesh = None
    
    def initialize(self):
        # Initialize Gmsh
        
    def finalize(self):
        # Finalize Gmsh
        
    def execute_code(self, code):
        # Execute Gmsh Python code
        # Collect statistics
        # Generate visualization
        # Return results
        
    def _collect_statistics(self):
        # Collect mesh statistics from Gmsh
        
    def _generate_visualization(self, mesh_data):
        # Generate visualization for the mesh
```

Key responsibilities:
- Initializing and finalizing Gmsh
- Executing Python code that uses the Gmsh API
- Collecting mesh statistics
- Generating mesh visualizations
- Loading and exporting meshes in different formats

### MeshData

The `MeshData` class stores all information about a generated mesh:

```python
class MeshData:
    def __init__(self, mesh_id=None):
        self.mesh_id = mesh_id or str(uuid.uuid4())
        self.nodes = []
        self.elements = []
        self.element_types = []
        self.boundaries = []
        self.physical_groups = {}
        self.statistics = {}
        self.output_file = None
        self.visualization_data = None
    
    def to_dict(self):
        # Convert mesh data to a dictionary
        
    def from_dict(self, data):
        # Load mesh data from a dictionary
        
    def save(self, filepath=None):
        # Save mesh data to a file
        
    @classmethod
    def load(cls, filepath):
        # Load mesh data from a file
```

Key responsibilities:
- Storing mesh geometry and topology
- Maintaining mesh statistics
- Tracking output files and visualization data
- Serializing and deserializing mesh data

## LLM Integration

### BaseLLM

The `BaseLLM` abstract base class defines the interface for all LLM providers:

```python
class BaseLLM(ABC):
    def __init__(self, model_name=None, temperature=0.2, max_tokens=2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def generate(self, prompt, system_prompt=None):
        # Generate a response from the LLM
        
    @abstractmethod
    async def generate_code(self, prompt, system_prompt=None):
        # Generate code from the LLM
        
    @abstractmethod
    async def chat(self, messages):
        # Have a multi-turn conversation with the LLM
        
    def extract_code_block(self, text):
        # Extract a Python code block from the LLM response
```

### LLM Providers

The system supports three LLM providers:

1. **OpenAILLM**: Integration with OpenAI's GPT models
2. **AnthropicLLM**: Integration with Anthropic's Claude models
3. **OllamaLLM**: Integration with local LLMs via Ollama

Each provider implements the `BaseLLM` interface with provider-specific API calls.

### LLMFactory

The `LLMFactory` class creates LLM instances based on the provider:

```python
class LLMFactory:
    @staticmethod
    def create_llm(provider, model_name=None, temperature=0.2, max_tokens=2048, **kwargs):
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAILLM(model_name=model_name, temperature=temperature, 
                             max_tokens=max_tokens, api_key=kwargs.get("api_key"))
        elif provider == "anthropic":
            return AnthropicLLM(model_name=model_name, temperature=temperature, 
                                max_tokens=max_tokens, api_key=kwargs.get("api_key"))
        elif provider == "ollama":
            return OllamaLLM(model_name=model_name, temperature=temperature, 
                             max_tokens=max_tokens, host=kwargs.get("host"))
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
```

## Communication Flow

The communication flow between components follows this sequence:

1. **User Input**: The user provides a natural language prompt describing the mesh they want to generate
2. **LLM Code Generation**: The LLM generates Python code using the Gmsh API based on the prompt
3. **Code Execution**: The Gmsh controller executes the generated code
4. **Mesh Creation**: Gmsh creates the mesh according to the code
5. **Statistics Collection**: The system collects statistics about the generated mesh
6. **Visualization**: The system generates a visualization of the mesh
7. **Result Return**: The mesh data is returned to the user

For refinement, the flow includes additional steps:
1. **User Feedback**: The user provides feedback on how to refine the mesh
2. **Context Preparation**: The system prepares context including mesh statistics
3. **LLM Refinement Code**: The LLM generates refinement code based on the feedback and context
4. **Refinement Execution**: The system executes the refinement code
5. **Updated Results**: The refined mesh data is returned to the user

## Mesh Generation Process

The mesh generation process is implemented in the `generate` method of the `MeshGenerator` class:

```python
async def generate(self, prompt, system_prompt=None):
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
```

The key steps are:
1. Generate Gmsh Python code from the natural language prompt using the LLM
2. Execute the generated code using the Gmsh controller
3. Store the conversation history for context
4. Return the generated mesh data

The code execution is handled by the `execute_code` method of the `GmshController` class:

```python
def execute_code(self, code):
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
```

## Iterative Refinement

The iterative refinement process is implemented in the `refine` method of the `MeshGenerator` class:

```python
async def refine(self, mesh_data, feedback):
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
```

The key aspects of the refinement process are:
1. **Context Preparation**: The system prepares a prompt with mesh statistics to provide context
2. **Feedback Incorporation**: The user's feedback is incorporated into the prompt
3. **Code Generation**: The LLM generates refinement code based on the context and feedback
4. **Execution**: The refinement code is executed to create a new, refined mesh
5. **Conversation Tracking**: The conversation history is updated to maintain context

This creates a feedback loop where:
- The user provides natural language feedback
- The LLM generates code to implement the feedback
- The system executes the code and provides updated results
- The user can provide further feedback based on the refined mesh

## Server Implementation

The server implementation uses FastAPI to provide a RESTful API for mesh generation and refinement:

```python
app = FastAPI(
    title="Gmsh MCP API",
    description="API for generating meshes using natural language prompts",
    version="0.1.0"
)

# Dictionary to store active mesh generators
mesh_generators: Dict[str, MeshGenerator] = {}

@app.post("/api/mesh/generate", response_model=MeshResponse)
async def generate_mesh(request: MeshRequest):
    # Create a mesh generator with the specified LLM config
    generator = MeshGenerator(
        llm_provider=request.llm_config.provider if request.llm_config else "openai",
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

@app.post("/api/mesh/refine", response_model=MeshResponse)
async def refine_mesh(request: MeshRefinementRequest):
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
```

The server also provides endpoints for:
- Chatting with the LLM about mesh generation
- Getting information about a mesh
- Downloading mesh files
- Getting mesh visualizations

## CLI Implementation

The CLI implementation provides a command-line interface for mesh generation and server management:

```python
def main():
    parser = argparse.ArgumentParser(description="Gmsh MCP Command-Line Interface")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate mesh command
    generate_parser = subparsers.add_parser("generate", help="Generate a mesh from a natural language prompt")
    generate_parser.add_argument("prompt", help="Natural language description of the mesh to generate")
    generate_parser.add_argument("--llm", dest="llm_provider", default="openai", choices=["openai", "anthropic", "ollama"], help="LLM provider to use")
    generate_parser.add_argument("--model", dest="model_name", help="Model name to use (provider-specific)")
    generate_parser.add_argument("--temperature", type=float, default=0.2, help="Controls randomness in the response (0.0 to 1.0)")
    generate_parser.add_argument("--format", dest="output_format", default="msh", help="Format to save the mesh in (msh, vtk, etc.)")
    generate_parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "generate":
        asyncio.run(generate_mesh(
            prompt=args.prompt,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            temperature=args.temperature,
            output_format=args.output_format,
            verbose=args.verbose
        ))
    elif args.command == "server":
        # Import here to avoid circular imports
        from ..server.main import run_server
        
        # Set environment variables for the server
        os.environ["SERVER_HOST"] = args.host
        os.environ["SERVER_PORT"] = str(args.port)
        os.environ["DEBUG"] = str(args.debug).lower()
        
        # Run the server
        run_server()
    else:
        parser.print_help()
```

## Data Models

The system uses Pydantic models for request and response validation:

```python
class LLMConfig(BaseModel):
    provider: str = Field(
        default="openai",
        description="LLM provider to use ('openai', 'anthropic', or 'ollama')"
    )
    model_name: Optional[str] = Field(
        default=None,
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
        default=None,
        description="Host URL for Ollama (if not using environment variables)"
    )

class MeshRequest(BaseModel):
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
```

## Visualization and Export

The system provides visualization and export capabilities:

```python
def _generate_visualization(self, mesh_data):
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

def export_mesh(self, mesh_data, format="msh"):
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
```

## Error Handling

The system implements error handling at multiple levels:

1. **LLM Error Handling**: Each LLM provider implements error handling for API calls
2. **Code Execution Error Handling**: The `execute_code` method captures and reports errors during code execution
3. **API Error Handling**: The FastAPI endpoints use try-except blocks to handle errors and return appropriate HTTP status codes
4. **CLI Error Handling**: The CLI commands handle errors and provide user-friendly error messages

Example of error handling in the API:

```python
@app.post("/api/mesh/generate", response_model=MeshResponse)
async def generate_mesh(request: MeshRequest):
    try:
        # Create a mesh generator and generate the mesh
        # ...
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Extension Points

The system is designed with several extension points:

1. **New LLM Providers**: Add a new class that implements the `BaseLLM` interface and update the `LLMFactory`
2. **Additional Mesh Operations**: Extend the `GmshController` class with new methods for specific mesh operations
3. **New Export Formats**: Add support for new export formats in the `export_mesh` method
4. **Custom Visualization**: Implement custom visualization methods in the `_generate_visualization` method
5. **Additional API Endpoints**: Add new endpoints to the FastAPI app for specific functionality
6. **New CLI Commands**: Add new subparsers and commands to the CLI

For example, to add a new LLM provider:

```python
class NewLLMProvider(BaseLLM):
    def __init__(self, model_name=None, temperature=0.2, max_tokens=2048, **kwargs):
        super().__init__(model_name, temperature, max_tokens)
        # Initialize provider-specific client
        
    async def generate(self, prompt, system_prompt=None):
        # Implement provider-specific generation
        
    async def generate_code(self, prompt, system_prompt=None):
        # Implement provider-specific code generation
        
    async def chat(self, messages):
        # Implement provider-specific chat
```

Then update the `LLMFactory`:

```python
@staticmethod
def create_llm(provider, model_name=None, temperature=0.2, max_tokens=2048, **kwargs):
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAILLM(...)
    elif provider == "anthropic":
        return AnthropicLLM(...)
    elif provider == "ollama":
        return OllamaLLM(...)
    elif provider == "new_provider":
        return NewLLMProvider(...)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
```

This modular design allows for easy extension and customization of the system to meet specific needs. 