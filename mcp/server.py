#!/usr/bin/env python3
"""
Horizon MCP Server Implementation

This module implements the Model Context Protocol (MCP) server for Horizon AI Assistant.
Protocol Version: 2025-06-18
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

# Core imports
from core import (
    setup_logging,
    get_ai_engine, get_personality_engine, get_memory_system, 
    get_media_engine, get_database_manager
)

logger = setup_logging("HorizonMCP")

class MCPServer:
    """Horizon MCP Protocol Server."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self.initialized = False
        
    def initialize(self):
        """Initialize core systems and register capabilities."""
        if self.initialized:
            return
            
        logger.info("🤖 Initializing Horizon MCP Server...")
        
        # Initialize core systems
        try:
            self.ai_engine = get_ai_engine()
            self.personality_engine = get_personality_engine()
            self.memory_system = get_memory_system()
            self.media_engine = get_media_engine()
            self.db_manager = get_database_manager()
            
            logger.info("✅ Core systems initialized")
        except Exception as e:
            logger.error(f"❌ Core system initialization failed: {e}")
            raise
        
        # Register tools
        self._register_tools()
        
        # Register resources
        self._register_resources()
        
        # Register prompts
        self._register_prompts()
        
        self.initialized = True
        logger.info(f"🚀 MCP Server ready: {len(self.tools)} tools, {len(self.resources)} resources, {len(self.prompts)} prompts")
    
    def _register_tools(self):
        """Register MCP tools."""
        self.tools = {
            "chat": {
                "name": "chat",
                "description": "Have a conversation with Horizon AI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to send"},
                        "personality": {"type": "string", "description": "Personality to use", "optional": True}
                    },
                    "required": ["message"]
                }
            },
            "generate_image": {
                "name": "generate_image",
                "description": "Generate an image using AI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Image description"},
                        "style": {"type": "string", "description": "Art style", "optional": True}
                    },
                    "required": ["prompt"]
                }
            },
            "generate_video": {
                "name": "generate_video",
                "description": "Generate a video using AI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Video description"},
                        "duration": {"type": "number", "description": "Duration in seconds", "optional": True}
                    },
                    "required": ["prompt"]
                }
            },
            "generate_audio": {
                "name": "generate_audio",
                "description": "Generate audio or music using AI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Audio description"},
                        "type": {"type": "string", "description": "Type: music, speech, sound", "optional": True}
                    },
                    "required": ["prompt"]
                }
            },
            "analyze_emotion": {
                "name": "analyze_emotion",
                "description": "Analyze emotion in text or image",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text or image path to analyze"},
                        "type": {"type": "string", "description": "Type: text or image", "optional": True}
                    },
                    "required": ["content"]
                }
            },
            "get_personality": {
                "name": "get_personality",
                "description": "Get information about a personality",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Personality name", "optional": True}
                    }
                }
            },
            "switch_personality": {
                "name": "switch_personality",
                "description": "Switch to a different personality",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Personality name"}
                    },
                    "required": ["name"]
                }
            },
            "get_memory_stats": {
                "name": "get_memory_stats",
                "description": "Get memory and learning statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def _register_resources(self):
        """Register MCP resources."""
        self.resources = {
            "chat_history": {
                "name": "chat_history",
                "description": "Recent chat conversation history",
                "mimeType": "application/json"
            },
            "personalities": {
                "name": "personalities",
                "description": "Available AI personalities",
                "mimeType": "application/json"
            },
            "memory_stats": {
                "name": "memory_stats",
                "description": "Memory and learning statistics",
                "mimeType": "application/json"
            }
        }
    
    def _register_prompts(self):
        """Register MCP prompts."""
        self.prompts = {
            "personality_prompts": {
                "name": "personality_prompts",
                "description": "Personality-based system prompts",
                "arguments": [
                    {
                        "name": "personality",
                        "description": "Personality name",
                        "required": True
                    }
                ]
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.initialized:
            self.initialize()
        return list(self.tools.values())
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources."""
        if not self.initialized:
            self.initialize()
        return list(self.resources.values())
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts."""
        if not self.initialized:
            self.initialize()
        return list(self.prompts.values())
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with arguments."""
        if not self.initialized:
            self.initialize()
            
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            if name == "chat":
                return await self._handle_chat(arguments)
            elif name == "generate_image":
                return await self._handle_generate_image(arguments)
            elif name == "generate_video":
                return await self._handle_generate_video(arguments)
            elif name == "generate_audio":
                return await self._handle_generate_audio(arguments)
            elif name == "analyze_emotion":
                return await self._handle_analyze_emotion(arguments)
            elif name == "get_personality":
                return await self._handle_get_personality(arguments)
            elif name == "switch_personality":
                return await self._handle_switch_personality(arguments)
            elif name == "get_memory_stats":
                return await self._handle_get_memory_stats(arguments)
            else:
                return {"error": f"Tool '{name}' not implemented"}
        except Exception as e:
            logger.error(f"Tool '{name}' error: {e}")
            return {"error": str(e)}
    
    async def get_resource(self, name: str) -> Dict[str, Any]:
        """Get a resource."""
        if not self.initialized:
            self.initialize()
            
        if name not in self.resources:
            return {"error": f"Resource '{name}' not found"}
        
        try:
            if name == "chat_history":
                return await self._get_chat_history()
            elif name == "personalities":
                return await self._get_personalities()
            elif name == "memory_stats":
                return await self._get_memory_stats_resource()
            else:
                return {"error": f"Resource '{name}' not implemented"}
        except Exception as e:
            logger.error(f"Resource '{name}' error: {e}")
            return {"error": str(e)}
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt."""
        if not self.initialized:
            self.initialize()
            
        if name not in self.prompts:
            return {"error": f"Prompt '{name}' not found"}
        
        try:
            if name == "personality_prompts":
                return await self._get_personality_prompt(arguments)
            else:
                return {"error": f"Prompt '{name}' not implemented"}
        except Exception as e:
            logger.error(f"Prompt '{name}' error: {e}")
            return {"error": str(e)}
    
    # Tool handlers
    async def _handle_chat(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat tool."""
        message = args.get("message", "")
        personality = args.get("personality")
        
        if personality:
            self.personality_engine.set_current_personality(personality)
        
        response = self.ai_engine.generate_response(message, use_context=True)
        
        return {
            "type": "text",
            "text": response
        }
    
    async def _handle_generate_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle image generation tool."""
        prompt = args.get("prompt", "")
        style = args.get("style", "realistic")
        
        result = self.media_engine.generate_image(prompt, style=style)
        
        if result.get("success"):
            return {
                "type": "image",
                "data": result.get("file_path"),
                "mimeType": "image/png"
            }
        else:
            return {"error": result.get("error", "Image generation failed")}
    
    async def _handle_generate_video(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle video generation tool."""
        prompt = args.get("prompt", "")
        duration = args.get("duration", 5)
        
        result = self.media_engine.generate_video(prompt, duration=duration)
        
        if result.get("success"):
            return {
                "type": "video",
                "data": result.get("file_path"),
                "mimeType": "video/mp4"
            }
        else:
            return {"error": result.get("error", "Video generation failed")}
    
    async def _handle_generate_audio(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle audio generation tool."""
        prompt = args.get("prompt", "")
        audio_type = args.get("type", "music")
        
        result = self.media_engine.generate_audio(prompt, audio_type=audio_type)
        
        if result.get("success"):
            return {
                "type": "audio",
                "data": result.get("file_path"),
                "mimeType": "audio/wav"
            }
        else:
            return {"error": result.get("error", "Audio generation failed")}
    
    async def _handle_analyze_emotion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emotion analysis tool."""
        content = args.get("content", "")
        content_type = args.get("type", "text")
        
        result = self.ai_engine.analyze_emotion(content, content_type=content_type)
        
        return {
            "type": "text",
            "text": json.dumps(result, indent=2)
        }
    
    async def _handle_get_personality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get personality tool."""
        name = args.get("name")
        
        if name:
            personality = self.personality_engine.get_personality(name)
        else:
            personality = self.personality_engine.get_current_personality()
        
        return {
            "type": "text",
            "text": json.dumps(personality, indent=2)
        }
    
    async def _handle_switch_personality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle switch personality tool."""
        name = args.get("name", "")
        
        success = self.personality_engine.set_current_personality(name)
        
        if success:
            return {
                "type": "text",
                "text": f"Switched to personality: {name}"
            }
        else:
            return {"error": f"Personality '{name}' not found"}
    
    async def _handle_get_memory_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get memory stats tool."""
        stats = self.memory_system.get_memory_stats()
        
        return {
            "type": "text",
            "text": json.dumps(stats, indent=2)
        }
    
    # Resource handlers
    async def _get_chat_history(self) -> Dict[str, Any]:
        """Get chat history resource."""
        history = self.memory_system.get_recent_conversations(limit=10)
        
        return {
            "contents": [
                {
                    "uri": "horizon://chat_history",
                    "mimeType": "application/json",
                    "text": json.dumps(history, indent=2)
                }
            ]
        }
    
    async def _get_personalities(self) -> Dict[str, Any]:
        """Get personalities resource."""
        personalities = self.personality_engine.get_available_personalities()
        
        return {
            "contents": [
                {
                    "uri": "horizon://personalities",
                    "mimeType": "application/json",
                    "text": json.dumps(personalities, indent=2)
                }
            ]
        }
    
    async def _get_memory_stats_resource(self) -> Dict[str, Any]:
        """Get memory stats resource."""
        stats = self.memory_system.get_memory_stats()
        
        return {
            "contents": [
                {
                    "uri": "horizon://memory_stats",
                    "mimeType": "application/json",
                    "text": json.dumps(stats, indent=2)
                }
            ]
        }
    
    # Prompt handlers
    async def _get_personality_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get personality prompt."""
        personality_name = args.get("personality", "default")
        personality = self.personality_engine.get_personality(personality_name)
        
        if not personality:
            return {"error": f"Personality '{personality_name}' not found"}
        
        prompt = f"""You are {personality.get('name', 'Horizon AI')}, an AI assistant with the following characteristics:

Personality: {personality.get('description', 'A helpful AI assistant')}

Communication Style: {personality.get('communication_style', 'Friendly and professional')}

Expertise: {', '.join(personality.get('expertise', ['General knowledge']))}

Approach conversations with this personality and maintain consistency throughout the interaction."""
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": {
                        "type": "text",
                        "text": prompt
                    }
                }
            ]
        }
    
    async def run(self):
        """Run the MCP server (stdio mode)."""
        logger.info("🤖 Starting Horizon MCP Server...")
        self.initialize()
        
        # Simple stdio-based MCP server
        logger.info("📡 MCP Server ready for stdio communication")
        
        try:
            while True:
                # In a real implementation, this would handle MCP protocol messages
                # For now, we'll just keep the server alive
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 MCP Server stopped")

def create_mcp_server() -> MCPServer:
    """Create and return a new MCP server instance."""
    return MCPServer()