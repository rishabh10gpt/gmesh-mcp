"""
Configuration module for the Gmsh MCP system.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(os.getenv("MESH_OUTPUT_DIR", os.path.join(BASE_DIR, "output")))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_NAME = OLLAMA_MODEL
# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Gmsh Configuration
GMSH_EXECUTABLE_PATH = os.getenv("GMSH_EXECUTABLE_PATH", "")
VISUALIZATION_ENABLED = os.getenv("VISUALIZATION_ENABLED", "true").lower() == "true"

# LLM Prompts
SYSTEM_PROMPT = """
You are a specialized assistant for generating Gmsh Python code based on natural language descriptions.
Your task is to convert user requests for mesh generation into valid Python code using the Gmsh API.
Focus on creating accurate, efficient, and well-structured meshes according to the user's specifications.
Always include proper initialization and finalization of Gmsh in your code.
"""

# Default mesh parameters
DEFAULT_MESH_SIZE = 0.1
DEFAULT_MESH_DIMENSION = 3
DEFAULT_MESH_ORDER = 2  # Second-order elements by default 

# New additions from the code block
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Mesh settings
DEFAULT_MIN_QUALITY = 0.3
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_OPTIMIZE = True

# Quality thresholds
QUALITY_WARNING_THRESHOLD = 0.7
QUALITY_ERROR_THRESHOLD = 0.3

# File extensions
MESH_EXTENSION = ".msh"
VISUALIZATION_EXTENSION = ".png" 