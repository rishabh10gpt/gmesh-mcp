"""
Main server module for the Gmsh MCP system.
"""

import os
import uvicorn
from .api import app
from ..utils.config import SERVER_HOST, SERVER_PORT, DEBUG


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.server.api:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=DEBUG
    )


if __name__ == "__main__":
    run_server() 