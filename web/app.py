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

def create_app(config_class=Config):
    """
    Create and configure the Flask application.
    
    Args:
        config_class: Configuration class to use
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Load configuration
    app.config.from_object(config_class)
    
    # Set up session configuration
    app.config['SECRET_KEY'] = getattr(Config, 'SECRET_KEY', 'horizon-ai-secret-key-change-in-production')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
    
    # Initialize core systems
    @app.before_first_request
    def initialize_core_systems():
        """Initialize core systems when the app starts."""
        logger.info("🚀 Initializing Horizon Web Application...")
        
        # Validate configuration
        config_status = validate_config()
        logger.info(f"📊 Configuration status: {config_status}")
        
        # Initialize core modules
        ai_engine = get_ai_engine()
        db_manager = get_database_manager()
        personality_engine = get_personality_engine()
        memory_system = get_memory_system()
        media_engine = get_media_engine()
        
        logger.info("✅ All core systems initialized successfully")
        
        # Log available features
        available_generators = media_engine.get_available_generators()
        logger.info(f"🎨 Available media generators: {available_generators}")
        
        api_keys = config_status.get('api_keys', {})
        enabled_apis = [api for api, enabled in api_keys.items() if enabled]
        logger.info(f"🔑 Enabled APIs: {enabled_apis}")
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return render_template('500.html'), 500
    
    # Register blueprints/routes
    from .routes import register_routes
    register_routes(app)
    
    logger.info("🌐 Horizon Web Application created successfully")
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