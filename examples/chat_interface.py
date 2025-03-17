"""
Example of using the Gmsh MCP system's chat interface.
"""

import os
import sys
import asyncio
import argparse

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core import MeshGenerator


async def chat_session(generator: MeshGenerator):
    """
    Run an interactive chat session with the mesh generator.
    
    Args:
        generator: Mesh generator instance
    """
    print("Welcome to the Gmsh MCP chat interface!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'generate: <prompt>' to generate a mesh.")
    print("Type 'refine: <feedback>' to refine the last generated mesh.")
    print()
    
    current_mesh = None
    
    while True:
        # Get user input
        user_input = input("> ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Check if the user wants to generate a mesh
        if user_input.lower().startswith("generate:"):
            prompt = user_input[len("generate:"):].strip()
            print(f"Generating mesh from prompt: {prompt}")
            
            try:
                current_mesh = await generator.generate(prompt)
                
                print(f"\nMesh generated successfully!")
                print(f"Mesh ID: {current_mesh.mesh_id}")
                print(f"Output file: {current_mesh.output_file}")
                print(f"\nNodes: {current_mesh.statistics.get('num_nodes', 'unknown')}")
                print(f"Elements: {current_mesh.statistics.get('num_elements', 'unknown')}")
                
                # Print visualization file if available
                if current_mesh.visualization_data:
                    print(f"\nVisualization file: {current_mesh.visualization_data}")
            
            except Exception as e:
                print(f"\nError generating mesh: {e}")
            
            continue
        
        # Check if the user wants to refine a mesh
        if user_input.lower().startswith("refine:"):
            if not current_mesh:
                print("No mesh to refine. Generate a mesh first.")
                continue
            
            feedback = user_input[len("refine:"):].strip()
            print(f"Refining mesh with feedback: {feedback}")
            
            try:
                refined_mesh = await generator.refine(current_mesh, feedback)
                current_mesh = refined_mesh
                
                print(f"\nMesh refined successfully!")
                print(f"Mesh ID: {current_mesh.mesh_id}")
                print(f"Output file: {current_mesh.output_file}")
                print(f"\nNodes: {current_mesh.statistics.get('num_nodes', 'unknown')}")
                print(f"Elements: {current_mesh.statistics.get('num_elements', 'unknown')}")
                
                # Print visualization file if available
                if current_mesh.visualization_data:
                    print(f"\nVisualization file: {current_mesh.visualization_data}")
            
            except Exception as e:
                print(f"\nError refining mesh: {e}")
            
            continue
        
        # Otherwise, chat with the LLM
        try:
            response = await generator.chat(user_input)
            print(f"\n{response}\n")
        
        except Exception as e:
            print(f"\nError: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Chat with the Gmsh MCP system")
    parser.add_argument("--llm", default="ollama", choices=["openai", "anthropic", "ollama"], help="LLM provider to use")
    parser.add_argument("--model", default=None, help="Model name to use (provider-specific)")
    args = parser.parse_args()
    
    # Create a mesh generator
    generator = MeshGenerator(
        llm_provider=args.llm,
        model_name=args.model
    )
    
    # Start the chat session
    await chat_session(generator)


if __name__ == "__main__":
    asyncio.run(main()) 