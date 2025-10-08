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

# Import AI engine components
from .ai_engine import AIEngine, get_ai_engine, ask_chatgpt, ask_ai_model, generate_fallback_response

# Import personality components
from .personality import (
    PersonalityEngine, EmotionAnalyzer, MoodDetector, PersonalityBlender,
    get_personality_engine, get_emotion_analyzer, get_mood_detector, get_personality_blender,
    get_personality_profile, analyze_emotion, detect_mood_from_text, blend_personalities
)

# Import database components
from .database import (
    DatabaseManager, UserManager, ConversationManager, MemoryManager, AnalyticsManager,
    get_database_manager, get_user_manager, get_conversation_manager, 
    get_memory_manager, get_analytics_manager,
    get_database_connection, init_database, backup_database
)

# Import media generator components
from .media_generator import (
    MediaEngine, ImageGenerator, VideoGenerator, AudioGenerator, ModelGenerator,
    EnhancedMediaEngine, LogoGenerator, Enhanced3DModelGenerator,
    get_media_engine, get_enhanced_media_engine, get_logo_generator, get_enhanced_3d_generator,
    get_image_generator, get_video_generator, get_audio_generator, get_model_generator,
    generate_image, generate_video, generate_audio, generate_3d_model, generate_logo_design
)

# Import memory system components
from .memory_system import (
    MemorySystem, UserMemory, ContextManager, LearningEngine,
    DatabaseMemorySystem, EnhancedMemorySystem,
    get_memory_system, get_context_manager, get_learning_engine, get_user_memory,
    get_enhanced_memory_system, get_database_memory_system,
    store_user_memory, get_user_context, learn_from_interaction, get_personalized_response,
    save_user_memory, retrieve_user_memory, save_conversation, 
    build_conversation_context, extract_learning_patterns, get_memory_stats
)

# Import utility components
from .utils import (
    CoreLogger, ConfigValidator, InputSanitizer, ResponseFormatter, 
    PerformanceMonitor, DataProcessor,
    setup_logging, validate_config, sanitize_input, format_response,
    generate_unique_id, measure_time, log_info, log_warning, log_error, log_debug
)

# Make key components easily accessible
__all__ = [
    # AI Engine
    'AIEngine', 'get_ai_engine', 'ask_chatgpt', 'ask_ai_model', 'generate_fallback_response',
    
    # Personality System
    'PersonalityEngine', 'EmotionAnalyzer', 'MoodDetector', 'PersonalityBlender',
    'get_personality_engine', 'get_emotion_analyzer', 'get_mood_detector', 'get_personality_blender',
    'get_personality_profile', 'analyze_emotion', 'detect_mood_from_text', 'blend_personalities',
    
    # Database Operations
    'DatabaseManager', 'UserManager', 'ConversationManager', 'MemoryManager', 'AnalyticsManager',
    'get_database_manager', 'get_user_manager', 'get_conversation_manager', 
    'get_memory_manager', 'get_analytics_manager',
    'get_database_connection', 'init_database', 'backup_database',
    
    # Media Generation
    'MediaEngine', 'ImageGenerator', 'VideoGenerator', 'AudioGenerator', 'ModelGenerator',
    'EnhancedMediaEngine', 'LogoGenerator', 'Enhanced3DModelGenerator',
    'get_media_engine', 'get_enhanced_media_engine', 'get_logo_generator', 'get_enhanced_3d_generator',
    'get_image_generator', 'get_video_generator', 'get_audio_generator', 'get_model_generator',
    'generate_image', 'generate_video', 'generate_audio', 'generate_3d_model', 'generate_logo_design',
    
    # Memory System
    'MemorySystem', 'UserMemory', 'ContextManager', 'LearningEngine',
    'DatabaseMemorySystem', 'EnhancedMemorySystem',
    'get_memory_system', 'get_context_manager', 'get_learning_engine', 'get_user_memory',
    'get_enhanced_memory_system', 'get_database_memory_system',
    'store_user_memory', 'get_user_context', 'learn_from_interaction', 'get_personalized_response',
    'save_user_memory', 'retrieve_user_memory', 'save_conversation', 
    'build_conversation_context', 'extract_learning_patterns', 'get_memory_stats',
    
    # Utilities
    'CoreLogger', 'ConfigValidator', 'InputSanitizer', 'ResponseFormatter', 
    'PerformanceMonitor', 'DataProcessor',
    'setup_logging', 'validate_config', 'sanitize_input', 'format_response',
    'generate_unique_id', 'measure_time', 'log_info', 'log_warning', 'log_error', 'log_debug'
]