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