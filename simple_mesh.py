"""
Simple example of using the Gmsh MCP system to generate a mesh.
"""

import sys
import asyncio
import argparse
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from src.core.mesh_generator import MeshGenerator
from src.utils.config import (
    DEFAULT_MESH_SIZE,
    DEFAULT_MIN_QUALITY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OPTIMIZE,
    OLLAMA_MODEL_NAME
)

def format_float(value, precision=3):
    """Format a float value with given precision, handling unknown values."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate a simple mesh using the Gmsh MCP system")
    parser.add_argument("--prompt", default="Create a tetrahedral mesh for a sphere with radius 1 and maximum element size 0.1", help="Natural language prompt for mesh generation")
    parser.add_argument("--provider", choices=["ollama", "anthropic", "openai"], default="ollama", help="LLM provider to use")
    parser.add_argument("--model", default=OLLAMA_MODEL_NAME, help="Model name to use")
    parser.add_argument("--feedback", help="Optional feedback for mesh refinement")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    try:
        print("Initializing mesh generator...")
        # Initialize mesh generator
        mesh_generator = MeshGenerator(
            llm_provider=args.provider,
            model_name=args.model
        )
        
        print(f"\nGenerating mesh for: {args.prompt}")
        print(f"Using provider: {args.provider}")
        if args.model:
            print(f"Using model: {args.model}")
        if args.feedback:
            print(f"Using feedback: {args.feedback}")
        
        # Generate mesh
        print("\nStarting mesh generation...")
        mesh_data = await mesh_generator.generate(
            prompt=args.prompt,
            feedback=args.feedback
        )
        
        # Print stdout output
        if hasattr(mesh_data, 'output') and mesh_data.output:
            print("\nGeneration Output:")
            print(mesh_data.output)
        
        if mesh_data and mesh_data.mesh_id:
            print(f"\nMesh generated successfully! ID: {mesh_data.mesh_id}")
            
            # Load and display mesh statistics
            print("\nMesh Statistics:")
            print(f"Nodes: {mesh_data.statistics.get('num_nodes', 'unknown')}")
            print(f"Elements: {mesh_data.statistics.get('num_elements', 'unknown')}")
            print(f"Average Quality: {format_float(mesh_data.statistics.get('average_quality', 'unknown'))}")
            print(f"Worst Quality: {format_float(mesh_data.statistics.get('worst_element_quality', 'unknown'))}")
            
            # Display quality warnings
            avg_quality = mesh_data.statistics.get('average_quality', 0)
            worst_quality = mesh_data.statistics.get('worst_element_quality', 0)
            
            if isinstance(avg_quality, (int, float)) and avg_quality < 0.7:
                print("\nWarning: Mesh quality is below recommended threshold (0.7)")
            if isinstance(worst_quality, (int, float)) and worst_quality < 0.3:
                print("\nError: Some elements have very poor quality (< 0.3)")
            
            # Display quality distribution
            if 'quality_distribution' in mesh_data.statistics:
                print("\nQuality Distribution:")
                for range_str, count in mesh_data.statistics['quality_distribution'].items():
                    print(f"  {range_str}: {count} elements")
            
            # Print element types
            if 'element_types' in mesh_data.statistics:
                print(f"\nElement types: {', '.join(mesh_data.statistics['element_types'])}")
            
            # Print visualization file if available
            if hasattr(mesh_data, 'visualization_data') and mesh_data.visualization_data:
                print(f"\nVisualization file: {mesh_data.visualization_data}")
                print("You can open this file to view the mesh.")
        else:
            print("\nFailed to generate mesh")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            print("\nTraceback:")
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 