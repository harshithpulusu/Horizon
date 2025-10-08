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
    get_ai_engine, get_personality_engine, get_enhanced_memory_system, 
    get_enhanced_media_engine, get_database_manager
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
            self.memory_system = get_enhanced_memory_system()
            self.media_engine = get_enhanced_media_engine()
            self.db_manager = get_database_manager()
            
            logger.info("✅ Enhanced core systems initialized")
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
            },
            # Enhanced Day 4 Tools
            "generate_logo": {
                "name": "generate_logo",
                "description": "Generate professional logo designs",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "brand_name": {"type": "string", "description": "Brand name for the logo"},
                        "industry": {"type": "string", "description": "Industry type"},
                        "style": {"type": "string", "description": "Logo style (modern, vintage, creative, etc.)", "optional": True}
                    },
                    "required": ["brand_name", "industry"]
                }
            },
            "generate_3d_model": {
                "name": "generate_3d_model",
                "description": "Generate 3D models from text descriptions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "3D model description"},
                        "style": {"type": "string", "description": "Model style (realistic, lowpoly, stylized)", "optional": True}
                    },
                    "required": ["prompt"]
                }
            },
            "batch_generate": {
                "name": "batch_generate",
                "description": "Generate multiple media items in batch",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "media_type": {"type": "string", "description": "Type of media (image, video, audio)"},
                        "prompts": {"type": "array", "items": {"type": "string"}, "description": "List of prompts"},
                        "params": {"type": "object", "description": "Generation parameters", "optional": True}
                    },
                    "required": ["media_type", "prompts"]
                }
            },
            "analyze_user_patterns": {
                "name": "analyze_user_patterns",
                "description": "Analyze user interaction patterns for personalization",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"}
                    },
                    "required": ["user_id"]
                }
            },
            "get_conversation_summary": {
                "name": "get_conversation_summary",
                "description": "Get summarized conversation context",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier", "optional": True},
                        "user_id": {"type": "string", "description": "User identifier", "optional": True},
                        "limit": {"type": "number", "description": "Number of conversations to analyze", "optional": True}
                    }
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
            },
            # Enhanced Day 4 Resources
            "user_patterns": {
                "name": "user_patterns",
                "description": "User interaction patterns and preferences",
                "mimeType": "application/json"
            },
            "generation_capabilities": {
                "name": "generation_capabilities",
                "description": "Available media generation capabilities",
                "mimeType": "application/json"
            },
            "performance_metrics": {
                "name": "performance_metrics",
                "description": "Agent performance and usage metrics",
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
            # Enhanced Day 4 Tool Handlers
            elif name == "generate_logo":
                return await self._handle_generate_logo(arguments)
            elif name == "generate_3d_model":
                return await self._handle_generate_3d_model(arguments)
            elif name == "batch_generate":
                return await self._handle_batch_generate(arguments)
            elif name == "analyze_user_patterns":
                return await self._handle_analyze_user_patterns(arguments)
            elif name == "get_conversation_summary":
                return await self._handle_get_conversation_summary(arguments)
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
            # Enhanced Day 4 Resource Handlers
            elif name == "user_patterns":
                return await self._get_user_patterns()
            elif name == "generation_capabilities":
                return await self._get_generation_capabilities()
            elif name == "performance_metrics":
                return await self._get_performance_metrics()
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
    
    # Enhanced Day 4 Tool Handlers
    async def _handle_generate_logo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logo generation tool."""
        brand_name = args.get("brand_name", "")
        industry = args.get("industry", "")
        style = args.get("style", "modern")
        
        result = self.media_engine.generate_logo(brand_name, industry, style)
        
        if result.get("success"):
            return {
                "type": "image",
                "data": result.get("file_path") or result.get("url"),
                "mimeType": "image/png",
                "metadata": {
                    "brand_name": brand_name,
                    "industry": industry,
                    "style": style,
                    "generated_at": result.get("generated_at")
                }
            }
        else:
            return {"error": result.get("error", "Logo generation failed")}
    
    async def _handle_generate_3d_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle 3D model generation tool."""
        prompt = args.get("prompt", "")
        style = args.get("style", "realistic")
        
        result = self.media_engine.generate_enhanced_3d_model(prompt, style)
        
        if result.get("success"):
            return {
                "type": "model",
                "data": result.get("file_path") or result.get("url"),
                "mimeType": "model/obj",
                "metadata": {
                    "prompt": prompt,
                    "style": style,
                    "model_type": result.get("model"),
                    "generated_at": result.get("generated_at")
                }
            }
        else:
            return {"error": result.get("error", "3D model generation failed")}
    
    async def _handle_batch_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch media generation tool."""
        media_type = args.get("media_type", "")
        prompts = args.get("prompts", [])
        params = args.get("params", {})
        
        if not prompts:
            return {"error": "No prompts provided for batch generation"}
        
        results = []
        for i, prompt in enumerate(prompts[:5]):  # Limit to 5 for safety
            try:
                if media_type == "image":
                    result = self.media_engine.generate_media('image', prompt, params)
                elif media_type == "video":
                    result = self.media_engine.generate_media('video', prompt, params)
                elif media_type == "audio":
                    result = self.media_engine.generate_media('audio', prompt, params)
                else:
                    result = {"success": False, "error": f"Unsupported media type: {media_type}"}
                
                results.append({
                    "index": i,
                    "prompt": prompt,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "prompt": prompt,
                    "result": {"success": False, "error": str(e)}
                })
        
        return {
            "type": "batch_results",
            "media_type": media_type,
            "total_processed": len(results),
            "results": results
        }
    
    async def _handle_analyze_user_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user pattern analysis tool."""
        user_id = args.get("user_id", "")
        
        if not user_id:
            return {"error": "User ID required for pattern analysis"}
        
        try:
            patterns = self.memory_system.analyze_user_patterns(user_id)
            return {
                "type": "text",
                "text": json.dumps(patterns, indent=2)
            }
        except Exception as e:
            return {"error": f"Pattern analysis failed: {str(e)}"}
    
    async def _handle_get_conversation_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversation summary tool."""
        session_id = args.get("session_id")
        user_id = args.get("user_id")
        limit = args.get("limit", 10)
        
        try:
            conversations = self.memory_system.get_recent_conversations(
                user_id=user_id, 
                session_id=session_id, 
                limit=limit
            )
            
            # Create summary
            if conversations:
                summary = {
                    "total_conversations": len(conversations),
                    "date_range": {
                        "earliest": conversations[-1]['timestamp'] if conversations else None,
                        "latest": conversations[0]['timestamp'] if conversations else None
                    },
                    "personalities_used": list(set(c['personality'] for c in conversations if c['personality'])),
                    "avg_sentiment": sum(c['sentiment_score'] for c in conversations if c['sentiment_score']) / len([c for c in conversations if c['sentiment_score']]) if any(c['sentiment_score'] for c in conversations) else 0,
                    "conversations": conversations
                }
            else:
                summary = {
                    "total_conversations": 0,
                    "message": "No conversations found"
                }
            
            return {
                "type": "text",
                "text": json.dumps(summary, indent=2)
            }
        except Exception as e:
            return {"error": f"Conversation summary failed: {str(e)}"}
    
    # Enhanced Day 4 Resource Handlers
    async def _get_user_patterns(self) -> Dict[str, Any]:
        """Get user interaction patterns resource."""
        try:
            # Get global pattern statistics
            stats = self.memory_system.get_memory_stats()
            return {
                "contents": [
                    {
                        "uri": "horizon://user_patterns",
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "global_stats": stats,
                            "pattern_types": [
                                "personality_preferences",
                                "interaction_frequency",
                                "topic_preferences",
                                "sentiment_patterns"
                            ]
                        }, indent=2)
                    }
                ]
            }
        except Exception as e:
            return {"error": f"User patterns resource failed: {str(e)}"}
    
    async def _get_generation_capabilities(self) -> Dict[str, Any]:
        """Get media generation capabilities resource."""
        try:
            capabilities = self.media_engine.get_generation_capabilities()
            return {
                "contents": [
                    {
                        "uri": "horizon://generation_capabilities",
                        "mimeType": "application/json",
                        "text": json.dumps(capabilities, indent=2)
                    }
                ]
            }
        except Exception as e:
            return {"error": f"Generation capabilities resource failed: {str(e)}"}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics resource."""
        try:
            # Get basic performance metrics
            metrics = {
                "tools_available": len(self.tools),
                "resources_available": len(self.resources),
                "prompts_available": len(self.prompts),
                "initialized": self.initialized,
                "core_systems": {
                    "ai_engine": self.ai_engine is not None,
                    "personality_engine": self.personality_engine is not None,
                    "memory_system": self.memory_system is not None,
                    "media_engine": self.media_engine is not None,
                    "database_manager": self.db_manager is not None
                },
                "server_version": "1.0.0-day4-enhanced"
            }
            
            return {
                "contents": [
                    {
                        "uri": "horizon://performance_metrics",
                        "mimeType": "application/json",
                        "text": json.dumps(metrics, indent=2)
                    }
                ]
            }
        except Exception as e:
            return {"error": f"Performance metrics resource failed: {str(e)}"}

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