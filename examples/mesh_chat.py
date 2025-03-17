"""
Chat interface for the Gmsh Mesh Generator.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
import json
import tempfile
import traceback
from datetime import datetime

# Add the parent directory to the Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.mesh_generator import MeshGenerator
from src.utils.config import OUTPUT_DIR
from src.core.llm_controller import LLMController

def check_api_keys(provider: str) -> bool:
    """Check if necessary API keys are set for the selected provider."""
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        return False
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        return False
    elif provider == "ollama" and not os.getenv("OLLAMA_HOST"):
        print("ERROR: OLLAMA_HOST environment variable is not set.")
        return False
    return True

def display_mesh_statistics(mesh_data):
    """Display mesh statistics in a formatted way."""
    if not mesh_data or not mesh_data.statistics:
        print("Warning: No mesh statistics available.")
        return
    
    stats = mesh_data.statistics
    
    # Display basic statistics
    print("\nMesh Statistics:")
    print("-" * 50)
    print(f"Number of Nodes: {stats.get('num_nodes', 'Unknown')}")
    print(f"Number of Elements: {stats.get('num_elements', 'Unknown')}")
    
    # Display quality metrics
    worst_quality = stats.get("worst_element_quality", 0)
    avg_quality = stats.get("average_quality", 0)
    print(f"\nQuality Metrics:")
    print(f"Worst Element Quality: {worst_quality:.3f}")
    print(f"Average Quality: {avg_quality:.3f}")
    
    if worst_quality < 0.3:
        print("\nWarning: Some elements have poor quality (below 0.3)")
    
    # Display bounding box if available
    if "bounding_box" in stats:
        bbox = stats["bounding_box"]
        print(f"\nBounding Box: {bbox['min']} to {bbox['max']}")
    
    # Display quality distribution
    if "quality_distribution" in stats:
        print("\nQuality Distribution:")
        quality_dist = stats["quality_distribution"]
        total_elements = sum(quality_dist.values())
        poor_quality = sum(count for range_str, count in quality_dist.items() 
                         if float(range_str.split('-')[0]) < 0.3)
        poor_percentage = (poor_quality / total_elements) * 100 if total_elements > 0 else 0
        
        for range_str, count in quality_dist.items():
            print(f"{range_str}: {count} elements")
        
        if poor_percentage > 10:
            print(f"\nWarning: {poor_percentage:.1f}% of elements have poor quality (below 0.3)")

def save_conversation_history(history, output_file):
    """Save conversation history to a file."""
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Chat interface for generating meshes using natural language descriptions")
    parser.add_argument("--provider", choices=LLMController.SUPPORTED_PROVIDERS, 
                       default=LLMController.DEFAULT_PROVIDER,
                       help="LLM provider to use")
    parser.add_argument("--model", help="Model name to use (provider-specific)")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum number of mesh generation iterations")
    parser.add_argument("--history-file", default="conversation_history.json",
                       help="File to save conversation history")
    
    args = parser.parse_args()
    
    # Check API keys
    if not check_api_keys(args.provider):
        sys.exit(1)
    
    # Initialize mesh generator
    mesh_generator = MeshGenerator(
        llm_provider=args.provider,
        model_name=args.model,
        max_iterations=args.max_iterations
    )
    
    print(f"\nChat interface initialized with {args.provider} provider and model {args.model}")
    print("Type 'quit' to exit, 'history' to show conversation history")
    print("=" * 50)
    
    conversation_history = []
    
    try:
        while True:
            # Get user input
            prompt = input("\nDescribe the mesh you want to generate: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'history':
                if conversation_history:
                    print("\nConversation History:")
                    print("-" * 50)
                    for entry in conversation_history:
                        print(f"\nPrompt: {entry['prompt']}")
                        print(f"Mesh ID: {entry['mesh_id']}")
                        print(f"Timestamp: {entry['timestamp']}")
                        print(f"Provider: {entry['provider']}")
                        print(f"Model: {entry['model']}")
                else:
                    print("\nNo conversation history yet.")
                continue
            
            if not prompt:
                print("Please enter a description of the mesh you want to generate.")
                continue
            
            print("\nGenerating mesh...")
            
            try:
                # Run mesh generation
                mesh_data = asyncio.run(mesh_generator.generate(prompt))
                
                if mesh_data and mesh_data.mesh_id:
                    mesh_file = Path(OUTPUT_DIR) / "meshes" / f"{mesh_data.mesh_id}.msh"
                    if mesh_file.exists():
                        print(f"\nMesh generated successfully! Saved to: {mesh_file}")
                        display_mesh_statistics(mesh_data)
                        
                        # Add to conversation history
                        conversation_history.append({
                            "prompt": prompt,
                            "mesh_id": mesh_data.mesh_id,
                            "timestamp": datetime.now().isoformat(),
                            "provider": args.provider,
                            "model": args.model
                        })
                        
                        # Save conversation history
                        save_conversation_history(conversation_history, args.history_file)
                    else:
                        print("\nError: Mesh file was not created successfully")
                else:
                    print("\nError: Mesh generation failed - no mesh data returned")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print(traceback.format_exc())
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        # Clean up any temporary files
        try:
            temp_dir = Path(tempfile.gettempdir())
            for temp_file in temp_dir.glob("gmsh_*.py"):
                temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

if __name__ == "__main__":
    main() 