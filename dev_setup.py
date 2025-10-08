#!/usr/bin/env python3
"""
Horizon Development    # Initialize database
    try:
        logger.info("🗄️ Initializing database...")
        from core import get_database_manager
        db_manager = get_database_manager()
        db_manager.init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")onment Setup Script

This script sets up the development environment for Horizon AI Assistant.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import setup_logging, validate_config, get_database_manager

def setup_development_environment():
    """Set up the development environment."""
    logger = setup_logging("HorizonDevSetup")
    
    logger.info("🛠️ Setting up Horizon development environment...")
    
    # Create necessary directories
    directories = [
        'logs',
        'static/generated_images',
        'static/generated_videos', 
        'static/generated_audio',
        'static/generated_music',
        'static/generated_3d_models',
        'static/generated_avatars',
        'static/generated_logos',
        'static/generated_designs',
        'static/generated_gifs',
        'backups'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    # Initialize database
    try:
        logger.info("🗄 Initializing database...")
        db_manager = get_database_manager()
        db_manager.init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    # Validate configuration
    logger.info("🔍 Validating configuration...")
    config_status = validate_config()
    
    # Check API keys
    api_keys = config_status.get('api_keys', {})
    for api, enabled in api_keys.items():
        status = "✅" if enabled else "❌"
        logger.info(f"{status} {api.upper()} API: {'Configured' if enabled else 'Not configured'}")
    
    # Check directories
    directories_status = config_status.get('directories', {})
    for directory, exists in directories_status.items():
        status = "✅" if exists else "❌"
        logger.info(f"{status} Directory {directory}: {'Exists' if exists else 'Missing'}")
    
    # System info
    system_info = config_status.get('system_info', {})
    if system_info.get('python_version'):
        logger.info(f"🐍 Python version: {system_info['python_version'].split()[0]}")
    if system_info.get('platform'):
        logger.info(f"💻 Platform: {system_info['platform']}")
    
    # Create sample .env file if it doesn't exist
    env_file = project_root / '.env'
    if not env_file.exists():
        logger.info("📝 Creating sample .env file...")
        env_content = """# Horizon AI Assistant Environment Variables

# OpenAI API (for ChatGPT and DALL-E)
OPENAI_API_KEY=your_openai_api_key_here

# Google AI APIs
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_project_id_here
GOOGLE_CLOUD_REGION=us-central1

# Replicate API (for video/audio generation)
REPLICATE_API_TOKEN=your_replicate_token_here

# Flask Configuration
SECRET_KEY=your_secret_key_here
DEBUG=True
HOST=0.0.0.0
PORT=5000

# Database Configuration
DATABASE_PATH=ai_memory.db
BACKUP_PATH=backups/

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# Memory System Configuration
MEMORY_RETENTION_DAYS=365
CONTEXT_WINDOW_SIZE=10
LEARNING_THRESHOLD=3
"""
        env_file.write_text(env_content)
        logger.info("✅ Sample .env file created")
        logger.info("🔑 Please update .env with your actual API keys")
    else:
        logger.info("📝 .env file already exists")
    
    # Test core modules
    logger.info("🧪 Testing core modules...")
    try:
        from core import (
            get_ai_engine, get_database_manager, get_personality_engine,
            get_memory_system, get_media_engine
        )
        
        # Test AI engine
        ai_engine = get_ai_engine()
        logger.info(f"✅ AI Engine: {'Available' if ai_engine.ai_model_available else 'Fallback mode'}")
        
        # Test personality engine
        personality_engine = get_personality_engine()
        personalities = personality_engine.get_available_personalities()
        logger.info(f"✅ Personality Engine: {len(personalities)} personalities available")
        
        # Test media engine
        media_engine = get_media_engine()
        generators = media_engine.get_available_generators()
        logger.info(f"✅ Media Engine: {len(generators)} generators available")
        
        # Test memory system
        memory_system = get_memory_system()
        logger.info("✅ Memory System: Operational")
        
    except Exception as e:
        logger.error(f"❌ Core module test failed: {e}")
    
    logger.info("🎉 Development environment setup complete!")
    logger.info("ℹ️ Next steps:")
    logger.info("   1. Update .env file with your API keys")
    logger.info("   2. Run: python scripts/start_web.py")
    logger.info("   3. Or run: python scripts/start_mcp.py")
    logger.info("   4. Or run: python scripts/start_both.py")

def main():
    """Main entry point."""
    setup_development_environment()

if __name__ == '__main__':
    main()
