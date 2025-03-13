# Gmsh Model Context Protocol (MCP)

A system that enables Large Language Models (LLMs) to interact directly with Gmsh for computational mesh generation through natural language prompts.

## Overview

This project implements a Model Context Protocol (MCP) system for Gmsh, a finite element mesh generator. It allows users to describe mesh requirements in natural language, which are then interpreted by an LLM to generate appropriate Gmsh commands. The system supports both API-based LLMs (like OpenAI's GPT and Anthropic's Claude) and local LLMs via Ollama.

## Features

- Natural language interface for mesh generation
- Support for both API-based LLMs and local LLMs via Ollama
- Real-time feedback and visualization of generated meshes
- Interactive refinement of meshes based on user feedback
- Support for various mesh types and configurations
- Export capabilities for common simulation formats

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/rishabh10gpt/gmesh-mcp.git
   cd gmesh-mcp
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit `.env` to add your API keys if using cloud-based LLMs.

## Usage

### Starting the server

```
python -m src.server.main
```

### Using the CLI

```
python -m src.cli.main "Create a tetrahedral mesh for a sphere with radius 1 and maximum element size 0.1"
```

### Using the API

```python
from gmesh_mcp import MeshGenerator

generator = MeshGenerator(llm_provider="openai")  # or "anthropic", "ollama"
mesh = generator.generate("Create a tetrahedral mesh for a sphere with radius 1")
mesh.visualize()
mesh.export("sphere.msh")
```

## Architecture

The system consists of several components:

1. **LLM Interface**: Handles communication with different LLM providers
2. **Gmsh Controller**: Manages Gmsh operations through its Python API
3. **Server**: Provides a web interface and API endpoints
4. **Feedback System**: Processes mesh statistics and visualization for iterative refinement

## Examples

See the `examples/` directory for sample use cases and demonstrations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by the MCP system for Blender by Siddharth Ahuja
- Built on the powerful Gmsh open-source mesh generator
- Leverages advancements in LLMs for natural language understanding 