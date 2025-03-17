"""
Mesh generator for the MCP system.
Integrates LLM and Gmsh controller to generate meshes from natural language prompts.
"""

import os
import json
import asyncio
import datetime
import uuid
import re
from typing import Dict, List, Optional, Any, Tuple, Union

from .gmsh_controller import GmshController, MeshData
from .llm_controller import LLMController
from ..utils.config import OUTPUT_DIR
from ..llm.factory import LLMFactory
from ..llm.base import BaseLLM
from ..utils.config import SYSTEM_PROMPT


class MeshGenerator:
    """Class for generating meshes using LLM and Gmsh."""
    
    def __init__(self, llm_provider="openai", model_name=None, max_iterations=3):
        """Initialize the mesh generator."""
        self.llm_controller = LLMController(llm_provider=llm_provider, model_name=model_name)
        self.gmsh_controller = GmshController()
        self.max_iterations = max_iterations
        self.conversation_history = []
    
    async def generate(self, prompt: str, feedback: Optional[str] = None, 
                     progress_callback: Optional[callable] = None) -> MeshData:
        """
        Generate a mesh from a natural language prompt.
        
        Args:
            prompt: Natural language prompt describing the mesh to generate
            feedback: Optional feedback for refinement
            progress_callback: Optional callback function for reporting progress
            
        Returns:
            MeshData object containing the generated mesh
        """
        mesh_id = str(uuid.uuid4())
        current_feedback = feedback
        
        for iteration in range(self.max_iterations):
            print(f"\nStarting mesh generation process...")
            print("=" * 50 + "\n")
            
            # Report progress if callback is provided
            if progress_callback:
                progress_callback(
                    status=f"Iteration {iteration+1}/{self.max_iterations}: Generating code", 
                    progress_value=25 + (iteration * 25 / self.max_iterations),
                    iteration=iteration+1
                )
            
            # Generate Gmsh code using LLM
            if iteration == 0 and not current_feedback:
                gmsh_code = await self.llm_controller.generate_code(prompt)
            else:
                # For subsequent iterations, use feedback or quality issues
                if current_feedback:
                    gmsh_code = await self.llm_controller.generate_code(prompt, current_feedback)
                else:
                    quality_issues = self._validate_mesh(mesh_data)
                    if quality_issues:
                        current_feedback = self._update_prompt_for_quality(prompt, quality_issues)
                        gmsh_code = await self.llm_controller.generate_code(prompt, current_feedback)
                    else:
                        break
            
            print("Generated Gmsh code:")
            print("-" * 19)
            print(gmsh_code)
            print("-" * 19 + "\n")
            
            # Report progress if callback is provided
            if progress_callback:
                progress_callback(
                    status=f"Iteration {iteration+1}/{self.max_iterations}: Executing code", 
                    progress_value=50 + (iteration * 25 / self.max_iterations),
                    iteration=iteration+1
                )
            
            # Execute the Gmsh code
            success, output = self.gmsh_controller.execute_code(gmsh_code, mesh_id)
            
            # Create MeshData object
            mesh_data = MeshData(
                mesh_id=mesh_id,
                output_file=os.path.join(self.gmsh_controller.meshes_dir, f"{mesh_id}.msh"),
                statistics={}
            )
            mesh_data.output = output
            # Store the generated Gmsh code in the mesh_data object
            mesh_data.gmsh_code = gmsh_code
            
            # Extract statistics from output
            self._extract_statistics_from_output(mesh_data)
            
            # Save the conversation for this iteration
            self._save_conversation(prompt, gmsh_code, mesh_data)
            
            # Report progress if callback is provided
            if progress_callback:
                progress_callback(
                    status=f"Iteration {iteration+1}/{self.max_iterations}: Validating mesh", 
                    progress_value=75 + (iteration * 25 / self.max_iterations),
                    iteration=iteration+1
                )
            
            # Check if mesh generation was successful
            if not success:
                print(f"\nError generating mesh: {output}")
                if iteration < self.max_iterations - 1:
                    print(f"\nRetrying... (Iteration {iteration + 2}/{self.max_iterations})")
                    current_feedback = output  # Pass error output as feedback
                    continue
                else:
                    print("\nWarning: Reached maximum iterations without success")
                    break
            
            # Validate mesh quality
            quality_issues = self._validate_mesh(mesh_data)
            if not quality_issues:
                print("\nMesh generation successful with good quality!")
                break
            
            if iteration < self.max_iterations - 1:
                print(f"\nRetrying with improved settings... (Iteration {iteration + 2}/{self.max_iterations})")
                if not current_feedback:
                    prompt = self._update_prompt_for_quality(prompt, quality_issues)
            else:
                print("\nWarning: Reached maximum iterations without achieving desired quality")
        
            # Final progress report
            if progress_callback:
                progress_callback(
                    status="Mesh generation complete", 
                    progress_value=100,
                    iteration=iteration+1
                )
            
        return mesh_data
    
    def _validate_mesh(self, mesh_data: MeshData) -> List[str]:
        """
        Validate a mesh and provide feedback.
        
        Args:
            mesh_data: Mesh data to validate
            
        Returns:
            List of quality issues, empty if mesh is good
        """
        issues = []
        
        # Check if mesh file exists
        if not mesh_data.output_file or not os.path.exists(mesh_data.output_file):
            issues.append("Mesh file was not generated successfully")
            return issues
        
        # Check if we have basic mesh statistics
        if not mesh_data.statistics.get("num_nodes") or not mesh_data.statistics.get("num_elements"):
            issues.append("Failed to extract basic mesh statistics")
            return issues
        
        # Check number of elements
        num_elements = mesh_data.statistics.get("num_elements", 0)
        if num_elements < 10:
            issues.append(f"Mesh has too few elements ({num_elements})")
        
        # Check element quality
        worst_quality = mesh_data.statistics.get("worst_element_quality", 0)
        avg_quality = mesh_data.statistics.get("average_quality", 0)
        
        if worst_quality < 0.1:
            issues.append(f"Some elements have very poor quality (minimum quality: {worst_quality:.3f})")
        if avg_quality < 0.5:
            issues.append(f"Average element quality is low ({avg_quality:.3f})")
        
        # Check quality distribution
        quality_dist = mesh_data.statistics.get("quality_distribution", {})
        if quality_dist:
            poor_elements = sum(count for range_str, count in quality_dist.items() 
                              if float(range_str.split('-')[1]) < 0.3)
            if poor_elements > 0:
                issues.append(f"There are {poor_elements} elements with quality below 0.3")
        
        return issues
    
    def _update_prompt_for_quality(self, original_prompt: str, quality_issues: List[str]) -> str:
        """
        Update the prompt based on quality issues to improve mesh generation.
        
        Args:
            original_prompt: Original mesh generation prompt
            quality_issues: List of quality issues identified
            
        Returns:
            Updated prompt for better mesh generation
        """
        if not quality_issues:
            return original_prompt
            
        # Categorize issues
        geometric_issues = []
        quality_issues_list = []
        other_issues = []
        
        for issue in quality_issues:
            if any(keyword in issue.lower() for keyword in ["center", "symmetric", "volume", "dimension", "region"]):
                geometric_issues.append(issue)
            elif any(keyword in issue.lower() for keyword in ["quality", "element"]):
                quality_issues_list.append(issue)
            else:
                other_issues.append(issue)
        
        # Create a feedback prompt that includes the original request and categorized issues
        feedback_sections = []
        
        if geometric_issues:
            feedback_sections.append("Geometric Issues:")
            feedback_sections.extend(f"- {issue}" for issue in geometric_issues)
        
        if quality_issues_list:
            feedback_sections.append("\nMesh Quality Issues:")
            feedback_sections.extend(f"- {issue}" for issue in quality_issues_list)
        
        if other_issues:
            feedback_sections.append("\nOther Issues:")
            feedback_sections.extend(f"- {issue}" for issue in other_issues)
        
        feedback = "\n".join(feedback_sections)
        
        updated_prompt = f"""
Previous attempt to create mesh with prompt: "{original_prompt}"
Generated the following feedback:
{feedback}

Please generate improved Gmsh code that addresses these issues.
Focus on:
1. Geometric Accuracy:
   - Center shapes at origin unless specified otherwise
   - Ensure proper dimensions and proportions
   - Complete boolean operations (holes, cuts) correctly
   - Maintain symmetry where appropriate
2. Mesh Quality:
   - Use appropriate mesh size controls
   - Add refinement where needed
   - Optimize element shapes
3. Implementation:
   - Use correct Gmsh API calls
   - Synchronize after geometric operations
   - Ensure proper cleanup
"""
        return updated_prompt
    
    def _analyze_mesh_quality(self, statistics: Dict[str, Any]) -> str:
        """
        Analyze mesh quality and provide feedback.
        
        Args:
            statistics: Mesh statistics dictionary
            
        Returns:
            Quality analysis feedback string
        """
        feedback = []
        
        # Check if we have valid statistics
        if not statistics:
            return "Unable to analyze mesh quality: No statistics available."
        
        # Check number of elements
        num_elements = statistics.get("num_elements")
        if num_elements:
            if num_elements < 100:
                feedback.append(f"The mesh is very coarse with only {num_elements} elements and requires refinement.")
            elif num_elements > 1000000:
                feedback.append(f"The mesh is very fine with {num_elements} elements and might be computationally expensive.")
            else:
                feedback.append(f"The mesh has {num_elements} elements.")
        else:
            feedback.append("Unable to determine the number of elements in the mesh.")
        
        # Check element quality
        worst_quality = statistics.get("worst_element_quality")
        avg_quality = statistics.get("average_quality")
        
        # Extract quality from stdout if available
        if not worst_quality or worst_quality == 0:
            # Try to extract from quality distribution
            quality_dist = statistics.get("quality_distribution", {})
            if quality_dist:
                min_range = min(float(range_str.split('-')[0]) for range_str in quality_dist.keys())
                worst_quality = min_range
        
        if not avg_quality or avg_quality == 0:
            # Try to calculate from quality distribution
            quality_dist = statistics.get("quality_distribution", {})
            if quality_dist:
                total_elements = sum(quality_dist.values())
                weighted_quality = 0
                for range_str, count in quality_dist.items():
                    min_qual, max_qual = map(float, range_str.split('-'))
                    avg_qual = (min_qual + max_qual) / 2
                    weighted_quality += avg_qual * count
                
                avg_quality = weighted_quality / total_elements if total_elements > 0 else 0
        
        if worst_quality:
            if worst_quality < 0.1:
                feedback.append(f"Critical: Some elements have very poor quality (minimum quality: {worst_quality:.3f}). The mesh requires improvement.")
            elif worst_quality < 0.2:
                feedback.append(f"Warning: Some elements have poor quality (minimum quality: {worst_quality:.3f}). Consider mesh refinement.")
            else:
                feedback.append(f"The minimum element quality is {worst_quality:.3f}, which is acceptable.")
        else:
            feedback.append("Unable to determine the minimum element quality.")
        
        if avg_quality:
            if avg_quality < 0.5:
                feedback.append(f"The average element quality ({avg_quality:.3f}) is low. Consider improving the mesh.")
            elif avg_quality > 0.8:
                feedback.append(f"The average element quality ({avg_quality:.3f}) is good.")
            else:
                feedback.append(f"The average element quality is {avg_quality:.3f}, which is acceptable.")
        else:
            feedback.append("Unable to determine the average element quality.")
        
        # Check quality distribution
        quality_dist = statistics.get("quality_distribution", {})
        if quality_dist:
            poor_elements = sum(count for range_str, count in quality_dist.items() 
                              if float(range_str.split('-')[1]) < 0.3)
            if poor_elements > 0:
                feedback.append(f"There are {poor_elements} elements with quality below 0.3 that require improvement.")
        
        return "\n".join(feedback) if feedback else "Mesh quality is acceptable."
    
    async def refine(self, mesh_data: MeshData, feedback: str) -> MeshData:
        """
        Refine a mesh based on feedback.
        
        Args:
            mesh_data: Mesh data to refine
            feedback: Natural language feedback on how to refine the mesh
            
        Returns:
            Refined mesh data
        """
        # Add the feedback to the conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"Refine the mesh with the following feedback: {feedback}"
        })
        
        # Generate a system prompt with mesh statistics and quality information
        stats_prompt = f"""
You are refining a mesh with the following statistics:
- Number of nodes: {mesh_data.statistics.get('num_nodes', 'unknown')}
- Number of elements: {mesh_data.statistics.get('num_elements', 'unknown')}
- Element types: {', '.join(mesh_data.statistics.get('element_types', ['unknown']))}
- Worst element quality: {mesh_data.statistics.get('worst_element_quality', 'unknown')}
- Average element quality: {mesh_data.statistics.get('average_quality', 'unknown')}

Quality feedback: {feedback}

Generate Gmsh Python code that loads the existing mesh and refines it according to the feedback.
Focus on:
1. Improving element quality where needed
2. Refining regions with poor quality elements
3. Maintaining good aspect ratios
4. Preserving geometric features

The code should:
1. Initialize Gmsh
2. Load the existing mesh
3. Apply appropriate refinement strategies
4. Optimize the mesh
5. Save the refined mesh
"""
        
        # Generate Gmsh code for refinement
        code = await self.llm_controller.generate_code(feedback, stats_prompt)
        
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
    
    async def chat(self, message: str) -> str:
        """
        Chat with the LLM about mesh generation.
        
        Args:
            message: User message
            
        Returns:
            LLM response
        """
        # Add the message to the conversation
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Get a response from the LLM
        response = await self.llm_controller.chat(self.conversation_history)
        
        # Add the response to the conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _save_conversation(self, prompt: str, gmsh_code: str, mesh_data: MeshData) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            prompt: The user's prompt
            gmsh_code: The generated Gmsh code
            mesh_data: The mesh data object
        """
        # Create a conversation object
        conversation = {
            "prompt": prompt,
            "gmsh_code": gmsh_code,
            "mesh_id": mesh_data.mesh_id,
            "output_file": mesh_data.output_file,
            "statistics": mesh_data.statistics,
            "timestamp": datetime.datetime.now().isoformat(),
            "output": mesh_data.output
        }
        
        # Save the conversation to a file in the conversations directory
        conversation_file = os.path.join(self.gmsh_controller.conversations_dir, f"{mesh_data.mesh_id}.json")
        with open(conversation_file, "w") as f:
            json.dump(conversation, f, indent=2)
    
    def load_mesh(self, filepath: str) -> MeshData:
        """
        Load a mesh from a file.
        
        Args:
            filepath: Path to the mesh file
            
        Returns:
            Loaded mesh data
        """
        return self.gmsh_controller.load_mesh(filepath)
    
    def export_mesh(self, mesh_data: MeshData, format: str = "msh") -> str:
        """
        Export a mesh to a file in the specified format.
        
        Args:
            mesh_data: Mesh data to export
            format: Export format (msh, vtk, etc.)
            
        Returns:
            Path to the exported file
        """
        return self.gmsh_controller.export_mesh(mesh_data, format)
    
    def _extract_statistics_from_output(self, mesh_data: MeshData) -> None:
        """
        Extract statistics from the output and populate the mesh_data object.
        
        Args:
            mesh_data: Mesh data to extract statistics from
        """
        if not hasattr(mesh_data, 'output') or not mesh_data.output:
            return
            
        # Extract node and element counts
        node_element_match = re.search(r"Info\s*:\s*(\d+)\s+nodes\s+(\d+)\s+elements", mesh_data.output)
        if node_element_match:
            mesh_data.statistics["num_nodes"] = int(node_element_match.group(1))
            mesh_data.statistics["num_elements"] = int(node_element_match.group(2))
            print(f"Extracted from output: {mesh_data.statistics['num_nodes']} nodes, {mesh_data.statistics['num_elements']} elements")
        
        # Extract quality metrics
        opt_match = re.search(r"Optimization starts.*?with worst = ([\d.]+) / average = ([\d.]+)", mesh_data.output, re.DOTALL)
        if opt_match:
            mesh_data.statistics["worst_element_quality"] = float(opt_match.group(1))
            mesh_data.statistics["average_quality"] = float(opt_match.group(2))
            print(f"Extracted quality metrics from output: worst={mesh_data.statistics['worst_element_quality']}, avg={mesh_data.statistics['average_quality']}")
        
        # Extract quality distribution
        quality_stats = {}
        quality_pattern = r"Info\s*:\s*([\d.]+)\s*<\s*quality\s*<\s*([\d.]+)\s*:\s*(\d+)\s*elements"
        for match in re.finditer(quality_pattern, mesh_data.output):
            min_qual, max_qual, count = match.groups()
            range_key = f"{min_qual}-{max_qual}"
            quality_stats[range_key] = int(count)
        
        if quality_stats:
            mesh_data.statistics["quality_distribution"] = quality_stats
            print(f"Extracted quality distribution with {len(quality_stats)} ranges")
            
        # Try to extract statistics from JSON if present in the output
        json_match = re.search(r"Mesh Statistics:\n({.*})", mesh_data.output, re.DOTALL)
        if json_match:
            try:
                json_stats = json.loads(json_match.group(1))
                for key, value in json_stats.items():
                    if key not in mesh_data.statistics:
                        mesh_data.statistics[key] = value
                print("Extracted statistics from JSON output")
            except json.JSONDecodeError:
                print("Failed to parse JSON statistics from output")