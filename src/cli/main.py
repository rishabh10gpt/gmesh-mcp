"""
Command-line interface for the Gmsh MCP system.
"""

import os
import sys
import argparse
import asyncio
from typing import Optional, Dict, Any
import json

from ..core.mesh_generator import MeshGenerator
from ..utils.config import OUTPUT_DIR


async def generate_mesh(
    prompt: str,
    llm_provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    output_format: str = "msh",
    verbose: bool = False
) -> None:
    """
    Generate a mesh from a natural language prompt.
    
    Args:
        prompt: Natural language description of the mesh to generate
        llm_provider: LLM provider to use ('openai', 'anthropic', or 'ollama')
        model_name: Model name to use (provider-specific)
        temperature: Controls randomness in the response (0.0 to 1.0)
        output_format: Format to save the mesh in (msh, vtk, etc.)
        verbose: Whether to print verbose output
    """
    # Create a mesh generator
    generator = MeshGenerator(
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature
    )
    
    # Generate the mesh
    print(f"Generating mesh from prompt: {prompt}")
    print(f"Using LLM provider: {llm_provider}")
    
    try:
        mesh_data = await generator.generate(prompt)
        
        print(f"\nMesh generated successfully!")
        print(f"Mesh ID: {mesh_data.mesh_id}")
        print(f"Output file: {mesh_data.output_file}")
        
        # Print mesh statistics
        if verbose:
            print("\nMesh statistics:")
            for key, value in mesh_data.statistics.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"\nNodes: {mesh_data.statistics.get('num_nodes', 'unknown')}")
            print(f"Elements: {mesh_data.statistics.get('num_elements', 'unknown')}")
        
        # Export the mesh if a different format is requested
        if output_format != "msh":
            export_file = generator.export_mesh(mesh_data, output_format)
            print(f"\nExported mesh to: {export_file}")
        
        # Print visualization file if available
        if mesh_data.visualization_data:
            print(f"\nVisualization file: {mesh_data.visualization_data}")
            print("You can open this file to view the mesh.")
    
    except Exception as e:
        print(f"\nError generating mesh: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
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


if __name__ == "__main__":
    main() 