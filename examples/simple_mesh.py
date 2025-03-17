"""
Example script demonstrating mesh generation with iterative improvement.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mesh_generator import MeshGenerator

async def main():
    parser = argparse.ArgumentParser(description="Generate a mesh from a natural language prompt")
    parser.add_argument("--llm", default="ollama", help="LLM provider to use")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Model name to use")
    parser.add_argument("--prompt", required=True, help="Natural language prompt describing the mesh")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of refinement iterations")
    parser.add_argument("--feedback", help="Optional feedback for mesh refinement")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize mesh generator
    generator = MeshGenerator(
        llm_provider=args.llm,
        model_name=args.model,
        max_iterations=args.max_iterations
    )
    
    try:
        # Generate mesh
        mesh_data = await generator.generate(args.prompt, args.feedback)
        
        # Print results
        print("\nMesh Generation Results:")
        print("=" * 50)
        print(f"Mesh ID: {mesh_data.mesh_id}")
        print(f"Output file: {mesh_data.output_file}")
        print("\nMesh Statistics:")
        print(f"Number of nodes: {mesh_data.statistics.get('num_nodes', 'unknown')}")
        print(f"Number of elements: {mesh_data.statistics.get('num_elements', 'unknown')}")
        print(f"Element types: {', '.join(mesh_data.statistics.get('element_types', ['unknown']))}")
        print(f"Worst element quality: {mesh_data.statistics.get('worst_element_quality', 'unknown')}")
        print(f"Average quality: {mesh_data.statistics.get('average_quality', 'unknown')}")
        
        if mesh_data.statistics.get("quality_distribution"):
            print("\nQuality Distribution:")
            for range_str, count in mesh_data.statistics["quality_distribution"].items():
                print(f"Quality {range_str}: {count} elements")
        
    except Exception as e:
        print(f"Error generating mesh: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 