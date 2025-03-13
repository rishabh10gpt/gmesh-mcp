"""
Simple example of using the Gmsh MCP system to generate a mesh.
"""

import os
import sys
import asyncio
import argparse

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core import MeshGenerator


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate a simple mesh using the Gmsh MCP system")
    parser.add_argument("--prompt", default="Create a tetrahedral mesh for a sphere with radius 1 and maximum element size 0.1", help="Natural language prompt for mesh generation")
    parser.add_argument("--llm", default="openai", choices=["openai", "anthropic", "ollama"], help="LLM provider to use")
    parser.add_argument("--model", default=None, help="Model name to use (provider-specific)")
    args = parser.parse_args()
    
    # Create a mesh generator
    generator = MeshGenerator(
        llm_provider=args.llm,
        model_name=args.model
    )
    
    # Generate the mesh
    print(f"Generating mesh from prompt: {args.prompt}")
    print(f"Using LLM provider: {args.llm}")
    
    try:
        mesh_data = await generator.generate(args.prompt)
        
        print(f"\nMesh generated successfully!")
        print(f"Mesh ID: {mesh_data.mesh_id}")
        print(f"Output file: {mesh_data.output_file}")
        print(f"\nNodes: {mesh_data.statistics.get('num_nodes', 'unknown')}")
        print(f"Elements: {mesh_data.statistics.get('num_elements', 'unknown')}")
        
        # Print visualization file if available
        if mesh_data.visualization_data:
            print(f"\nVisualization file: {mesh_data.visualization_data}")
            print("You can open this file to view the mesh.")
    
    except Exception as e:
        print(f"\nError generating mesh: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 