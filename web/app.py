"""
Horizon Web Application

This is the refactored Flask application that uses the shared core module
for all business logic while providing the web interface.
"""

import os
import sys
from flask import Flask, request, session, render_template
from datetime import timedelta
from config import Config

# Add project root to path for core imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core modules
from core import (
    get_ai_engine, get_database_manager, get_personality_engine,
    get_memory_system, get_media_engine, setup_logging, validate_config
)

# Initialize logging
logger = setup_logging("HorizonWeb")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure app
    app.config.from_object(Config)
    
    # Initialize core systems
    def init_core_systems():
        """Initialize core systems."""
        try:
            logger.info("� Initializing core systems...")
            
            # Initialize database
            db_manager = get_database_manager()
            db_manager.init_database()
            
            # Initialize AI engine
            ai_engine = get_ai_engine()
            logger.info(f"🤖 AI Engine status: {'Ready' if ai_engine.ai_model_available else 'Fallback mode'}")
            
            # Initialize personality engine
            personality_engine = get_personality_engine()
            personalities = personality_engine.get_available_personalities()
            logger.info(f"🎭 Loaded {len(personalities)} personalities")
            
            # Initialize memory system
            memory_system = get_memory_system()
            logger.info("🧠 Memory system initialized")
            
            # Initialize media engine
            media_engine = get_media_engine()
            generators = media_engine.get_available_generators()
            logger.info(f"🎨 Media engine: {len(generators)} generators available")
            
            logger.info("✅ Core systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Core system initialization failed: {e}")
    
    # Call initialization immediately for modern Flask
    with app.app_context():
        init_core_systems()
    
    # Register routes
    from web.routes import register_routes
    register_routes(app)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"404 error: {request.url}")
        return render_template('error.html', 
                             error_code=404, 
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        return render_template('error.html', 
                             error_code=500, 
                             error_message="Internal server error"), 500
    
    return app

def get_app():
    """Get the Flask application instance."""
    return create_app()

# Create the app instance
app = create_app()

if __name__ == '__main__':
    # Development server
    app.run(
        host=getattr(Config, 'HOST', '0.0.0.0'),
        port=getattr(Config, 'PORT', 5000),
        debug=getattr(Config, 'DEBUG', True)
    )