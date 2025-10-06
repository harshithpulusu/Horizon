"""
Horizon Core Module

This module contains the shared business logic for Horizon AI Assistant.
It provides core functionality that can be used by both the web interface
and the MCP agent.

Modules:
- ai_engine: AI model integrations (ChatGPT, Gemini)
- personality: Personality system and blending
- database: Database operations and management
- media_generator: Image, video, and audio generation
- memory_system: User memory and context management
- utils: Shared utilities and helpers
"""

__version__ = "1.0.0"
__author__ = "Horizon AI Team"

# Import AI engine components for easy access
from .ai_engine import AIEngine, get_ai_engine, ask_chatgpt, ask_ai_model, generate_fallback_response

# Make AI engine easily accessible
__all__ = [
    'AIEngine',
    'get_ai_engine', 
    'ask_chatgpt',
    'ask_ai_model',
    'generate_fallback_response'
]