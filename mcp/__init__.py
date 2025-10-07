"""
Horizon MCP Agent Module

This module implements the Model Context Protocol (MCP) server for Horizon AI Assistant.
It exposes Horizon's capabilities as tools, resources, and prompts that can be used
by other AI systems.

Current Protocol Version: 2025-06-18 (latest MCP specification)

Modules:
- server: Main MCP server implementation
- tools: Tool definitions and implementations
- resources: Resource handlers for data access
- prompts: Personality-based prompt definitions
- schemas: Pydantic schemas for data validation

Features:
- Image, video, and audio generation tools
- Personality blending and analysis tools
- Conversation and context management
- User memory operations
- Predictive assistance tools
"""

__version__ = "1.0.0"
__author__ = "Horizon AI Team"
__mcp_version__ = "2025-06-18"

# MCP module imports
try:
    from .server import create_mcp_server, MCPServer
    __all__ = ['create_mcp_server', 'MCPServer']
except ImportError as e:
    # Modules not yet created or have issues
    print(f"MCP import warning: {e}")
    __all__ = []

# Add placeholder functions for the other imports that might fail
try:
    from .tools import register_tools
    __all__.append('register_tools')
except ImportError:
    pass

try:
    from .resources import register_resources
    __all__.append('register_resources')
except ImportError:
    pass

try:
    from .prompts import register_prompts
    __all__.append('register_prompts')
except ImportError:
    pass