"""
Server module for the Gmsh MCP system.
"""

from .api import app
from .main import run_server

__all__ = [
    "app",
    "run_server"
] 