"""
Streamlit-based web interface for the Gmsh MCP system.
Provides an interactive UI with tunable parameters for mesh generation.
"""

import sys
import asyncio
import traceback
from pathlib import Path
import time
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Try to import plotly for gauge charts
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.core.mesh_generator import MeshGenerator
from src.utils.config import (
    DEFAULT_MESH_SIZE,
    DEFAULT_MIN_QUALITY,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OPTIMIZE,
    OLLAMA_MODEL_NAME,
    QUALITY_WARNING_THRESHOLD,
    QUALITY_ERROR_THRESHOLD,
    OUTPUT_DIR
)

# Set page configuration
st.set_page_config(
    page_title="Gmsh MCP - Mesh Generation",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define async function to run in sync context
async def generate_mesh_async(prompt, provider, model, mesh_size_factor, max_iterations, optimize):
    """Generate a mesh asynchronously."""
    try:
        # Initialize mesh generator
        mesh_generator = MeshGenerator(
            llm_provider=provider,
            model_name=model,
            max_iterations=max_iterations
        )
        
        # Add mesh size and optimization parameters to the prompt
        enhanced_prompt = f"{prompt}\n\nUse a mesh size factor of {mesh_size_factor} and {'enable' if optimize else 'disable'} mesh optimization."
        
        # Create a custom progress callback
        progress_data = {"status": "Initializing", "progress": 0, "current_iteration": 0}
        
        # Define a progress callback
        def progress_callback(status, progress_value=None, iteration=None):
            progress_data["status"] = status
            if progress_value is not None:
                progress_data["progress"] = progress_value
            if iteration is not None:
                progress_data["current_iteration"] = iteration
        
        # Generate mesh with progress reporting
        progress_callback("Generating Gmsh code", 25, 1)
        mesh_data = await mesh_generator.generate(
            prompt=enhanced_prompt,
            progress_callback=progress_callback
        )
        
        # Set final progress
        progress_callback("Mesh generation complete", 100)
        
        return mesh_data, progress_data
    except Exception as e:
        st.error(f"Error generating mesh: {str(e)}")
        traceback.print_exc()
        return None, {"status": "Error", "progress": 0, "current_iteration": 0}

def run_async(coroutine):
    """Run an async function in a synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

def display_mesh_statistics(mesh_data):
    """Display mesh statistics in a formatted way."""
    if not mesh_data or not mesh_data.statistics:
        st.warning("No mesh statistics available.")
        return
    
    stats = mesh_data.statistics
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", stats.get("num_nodes", "Unknown"))
    with col2:
        st.metric("Elements", stats.get("num_elements", "Unknown"))
    with col3:
        element_types = stats.get("element_types", ["Unknown"])
        st.metric("Element Type", ", ".join(element_types))
    
    # Quality metrics with gauge charts
    st.subheader("Mesh Quality")
    
    # Create gauge charts for quality metrics
    worst_quality = stats.get("worst_element_quality", 0)
    avg_quality = stats.get("average_quality", 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Worst quality gauge
        fig = create_gauge_chart(
            value=worst_quality, 
            title="Worst Element Quality",
            threshold_poor=QUALITY_ERROR_THRESHOLD,
            threshold_fair=QUALITY_WARNING_THRESHOLD
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average quality gauge
        fig = create_gauge_chart(
            value=avg_quality, 
            title="Average Element Quality",
            threshold_poor=QUALITY_ERROR_THRESHOLD,
            threshold_fair=QUALITY_WARNING_THRESHOLD
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Quality distribution
    quality_dist = stats.get("quality_distribution", {})
    if quality_dist:
        st.subheader("Quality Distribution")
        
        # Convert to DataFrame for display
        dist_data = []
        for range_str, count in quality_dist.items():
            min_qual, max_qual = map(float, range_str.split('-'))
            dist_data.append({
                "Quality Range": f"{min_qual:.1f} - {max_qual:.1f}",
                "Count": count,
                "Min": min_qual,
                "Max": max_qual
            })
        
        df = pd.DataFrame(dist_data)
        df = df.sort_values(by="Min")
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df["Quality Range"], df["Count"], color='skyblue')
        
        # Color bars based on quality thresholds
        for i, bar in enumerate(bars):
            if df.iloc[i]["Max"] < QUALITY_ERROR_THRESHOLD:
                bar.set_color('red')
            elif df.iloc[i]["Max"] < QUALITY_WARNING_THRESHOLD:
                bar.set_color('orange')
        
        ax.set_xlabel("Element Quality")
        ax.set_ylabel("Number of Elements")
        ax.set_title("Element Quality Distribution")
        plt.xticks(rotation=45)
        
        # Display the chart
        st.pyplot(fig)
        
        # Display as table
        st.dataframe(df[["Quality Range", "Count"]])

def create_gauge_chart(value, title, threshold_poor=0.3, threshold_fair=0.7):
    """Create a gauge chart for quality metrics."""
    try:
        # Define colors for different quality ranges
        if value < threshold_poor:
            color = "red"
        elif value < threshold_fair:
            color = "orange"
        else:
            color = "green"
        
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold_poor], 'color': 'rgba(255, 0, 0, 0.2)'},
                    {'range': [threshold_poor, threshold_fair], 'color': 'rgba(255, 165, 0, 0.2)'},
                    {'range': [threshold_fair, 1], 'color': 'rgba(0, 128, 0, 0.2)'}
                ],
            }
        ))
        
        # Update layout
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=50, b=10),
            font={'size': 12}
        )
        
        return fig
    except ImportError:
        # Fallback if plotly is not available
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(4, 0.8))
        ax.barh(0, value, height=0.4, color='blue')
        ax.barh(0, 1, height=0.4, color='lightgray', alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)
        ax.text(value, 0, f"{value:.3f}", va='center', ha='center', fontweight='bold')
        ax.set_yticks([])
        ax.set_xticks([0, threshold_poor, threshold_fair, 1])
        ax.set_xticklabels(['0', str(threshold_poor), str(threshold_fair), '1'])
        
        # Add colored regions
        ax.axvspan(0, threshold_poor, color='red', alpha=0.2)
        ax.axvspan(threshold_poor, threshold_fair, color='orange', alpha=0.2)
        ax.axvspan(threshold_fair, 1, color='green', alpha=0.2)
        
        return fig

def display_mesh_visualization(mesh_data):
    """Display mesh visualization if available."""
    if not mesh_data:
        return
    
    # Check if visualization data is available
    vis_file = mesh_data.visualization_data if hasattr(mesh_data, 'visualization_data') else None
    
    if vis_file and os.path.exists(vis_file):
        st.subheader("Mesh Visualization")
        try:
            image = Image.open(vis_file)
            st.image(image, caption="Mesh Visualization", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying visualization: {str(e)}")
    else:
        # If no visualization is available, show a placeholder
        st.subheader("Mesh Visualization")
        st.info("No visualization available for this mesh.")

def display_mesh_output(mesh_data):
    """Display mesh generation output."""
    if not mesh_data:
        return
    
    output = mesh_data.output if hasattr(mesh_data, 'output') else None
    
    if output:
        with st.expander("Mesh Generation Output", expanded=False):
            st.code(output, language="bash")

def main():
    """Main function for the Streamlit app."""
    st.title("Gmsh MCP - Mesh Generation")
    st.markdown("""
    Generate meshes from natural language descriptions using the Gmsh MCP system.
    Adjust parameters to control mesh generation and quality.
    """)
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # LLM provider selection
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=["ollama", "openai", "anthropic"],
        index=0
    )
    
    # Model selection based on provider
    if provider == "ollama":
        model = st.sidebar.text_input("Model Name", value=OLLAMA_MODEL_NAME)
    elif provider == "openai":
        model = st.sidebar.selectbox(
            "Model",
            options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
    elif provider == "anthropic":
        model = st.sidebar.selectbox(
            "Model",
            options=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            index=1
        )
    
    # Mesh generation parameters
    st.sidebar.subheader("Mesh Parameters")
    
    mesh_size_factor = st.sidebar.slider(
        "Mesh Size Factor",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Controls the size of mesh elements. Smaller values create finer meshes."
    )
    
    max_iterations = st.sidebar.slider(
        "Maximum Iterations",
        min_value=1,
        max_value=10,
        value=DEFAULT_MAX_ITERATIONS,
        step=1,
        help="Maximum number of iterations for mesh generation."
    )
    
    optimize = st.sidebar.checkbox(
        "Optimize Mesh",
        value=DEFAULT_OPTIMIZE,
        help="Enable mesh optimization to improve element quality."
    )
    
    # Advanced options in expander
    with st.sidebar.expander("Advanced Options"):
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_MIN_QUALITY,
            step=0.05,
            help="Minimum acceptable element quality."
        )
        
        visualization = st.checkbox(
            "Generate Visualization",
            value=True,
            help="Generate and display mesh visualization."
        )
    
    # Main content area
    st.subheader("Mesh Description")
    
    # Example prompts
    example_prompts = [
        "Create a tetrahedral mesh for a cube with side length 5",
        "Generate a mesh for a sphere with radius 2 centered at the origin",
        "Create a tetrahedral mesh for a cube with side length 5 with a cylindrical hole of radius 1 through the center",
        "Generate a mesh for a mechanical part consisting of a rectangular base (10x5x1) with a cylindrical pillar (radius 1, height 3) in the center"
    ]
    
    selected_example = st.selectbox(
        "Example Prompts",
        options=["Select an example prompt..."] + example_prompts,
        index=0
    )
    
    prompt = st.text_area(
        "Enter your mesh description",
        value=selected_example if selected_example != "Select an example prompt..." else "",
        height=100,
        help="Describe the mesh you want to generate in natural language."
    )
    
    # Generate button
    generate_button = st.button("Generate Mesh", type="primary", disabled=not prompt)
    
    if generate_button and prompt:
        with st.spinner("Generating mesh..."):
            # Create a progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Initializing mesh generation...")
            
            # Show a placeholder for the code
            code_placeholder = st.empty()
            
            # Generate the mesh
            mesh_data, progress_data = run_async(generate_mesh_async(
                prompt=prompt,
                provider=provider,
                model=model,
                mesh_size_factor=mesh_size_factor,
                max_iterations=max_iterations,
                optimize=optimize
            ))
            
            # Update progress bar and status text
            progress_bar.progress(progress_data["progress"])
            status_text.info(f"Status: {progress_data['status']}")
            
            # Clear the placeholder
            code_placeholder.empty()
            
            if mesh_data:
                st.success(f"Mesh generated successfully! ID: {mesh_data.mesh_id}")
                
                # Display the generated code in a collapsible section
                if hasattr(mesh_data, 'gmsh_code'):
                    with st.expander("Generated Gmsh Code", expanded=False):
                        st.code(mesh_data.gmsh_code, language="python")
                
                # Display tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Statistics", "Visualization", "Output", "Stdout"])
                
                with tab1:
                    display_mesh_statistics(mesh_data)
                
                with tab2:
                    display_mesh_visualization(mesh_data)
                
                with tab3:
                    # Display the output file path
                    if mesh_data.output_file:
                        st.info(f"Output file: {mesh_data.output_file}")
                    
                    # Display mesh output in a collapsible section
                    display_mesh_output(mesh_data)
                
                with tab4:
                    # Display raw stdout/stderr
                    if hasattr(mesh_data, 'output'):
                        st.subheader("Standard Output/Error")
                        st.text_area("Raw Output", value=mesh_data.output, height=400, disabled=True)
                
                # Download button for the mesh file
                if mesh_data.output_file and os.path.exists(mesh_data.output_file):
                    with open(mesh_data.output_file, "rb") as f:
                        mesh_bytes = f.read()
                    
                    st.download_button(
                        label="Download Mesh File",
                        data=mesh_bytes,
                        file_name=os.path.basename(mesh_data.output_file),
                        mime="application/octet-stream"
                    )
            else:
                st.error("Failed to generate mesh. Please check the logs for details.")
    
    # Display recent meshes
    st.subheader("Recent Meshes")
    
    # Get list of mesh files
    mesh_dir = os.path.join(OUTPUT_DIR, "meshes")
    if os.path.exists(mesh_dir):
        mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".msh")]
        mesh_files.sort(key=lambda x: os.path.getmtime(os.path.join(mesh_dir, x)), reverse=True)
        
        if mesh_files:
            # Display as a table
            mesh_data = []
            for mesh_file in mesh_files[:10]:  # Show only the 10 most recent
                file_path = os.path.join(mesh_dir, mesh_file)
                mesh_id = os.path.splitext(mesh_file)[0]
                created_time = time.ctime(os.path.getmtime(file_path))
                size_kb = os.path.getsize(file_path) / 1024
                
                mesh_data.append({
                    "Mesh ID": mesh_id,
                    "Created": created_time,
                    "Size (KB)": f"{size_kb:.2f}"
                })
            
            df = pd.DataFrame(mesh_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No meshes generated yet.")
    else:
        st.info("Mesh directory not found.")

if __name__ == "__main__":
    main() 