# Gmsh MCP Examples

This directory contains example scripts demonstrating how to use the Gmsh MCP system.

## Simple Mesh Generation

The `simple_mesh.py` script demonstrates how to generate a mesh from a natural language prompt:

```bash
python simple_mesh.py --prompt "Create a tetrahedral mesh for a sphere with radius 1" --llm openai
```

## Mesh Refinement

The `mesh_refinement.py` script demonstrates how to generate and refine a mesh:

```bash
python mesh_refinement.py --prompt "Create a tetrahedral mesh for a cube with side length 1" --feedback "Refine the mesh near the corners with smaller elements" --llm openai
```

## Chat Interface

The `chat_interface.py` script provides an interactive chat interface for mesh generation and refinement:

```bash
python chat_interface.py --llm openai
```

In the chat interface, you can:
- Type `generate: <prompt>` to generate a mesh
- Type `refine: <feedback>` to refine the last generated mesh
- Type any other text to chat with the LLM about mesh generation
- Type `exit` or `quit` to end the session

## Using Different LLM Providers

All examples support different LLM providers:

- OpenAI (default):
  ```bash
  python simple_mesh.py --llm openai --model gpt-4o
  ```

- Anthropic:
  ```bash
  python simple_mesh.py --llm anthropic --model claude-3-opus-20240229
  ```

- Ollama (local LLM):
  ```bash
  python simple_mesh.py --llm ollama --model llama3
  ```

Make sure to set up your API keys in the `.env` file before using API-based LLMs. 