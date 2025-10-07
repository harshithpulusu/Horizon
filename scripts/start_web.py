#!/usr/bin/env python3
"""
Horizon Web Server Startup Script

This script starts the Flask web server for Horizon AI Assistant.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import web module
from web import create_app
from core import setup_logging, validate_config
from config import Config

def start_web_server(host=None, port=None, debug=None):
    """Start the web server."""
    logger = setup_logging("HorizonWebServer")
    
    # Set defaults from config
    host = host or getattr(Config, 'HOST', '0.0.0.0')
    port = port or getattr(Config, 'PORT', 5000)
    debug = debug if debug is not None else getattr(Config, 'DEBUG', True)
    
    logger.info("🌐 Starting Horizon Web Server...")
    logger.info(f"🔗 Server will run on http://{host}:{port}")
    
    # Validate configuration
    config_status = validate_config()
    logger.info(f"📊 Configuration status: {config_status}")
    
    # Create and run the Flask app
    app = create_app()
    
    try:
        logger.info("✅ Web server starting...")
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("🚪 Web server stopped by user")
    except Exception as e:
        logger.error(f"❌ Web server error: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Start Horizon Web Server')
    parser.add_argument('--host', default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    args = parser.parse_args()
    
    debug = None
    if args.debug:
        debug = True
    elif args.no_debug:
        debug = False
    
    start_web_server(args.host, args.port, debug)

if __name__ == '__main__':
    main()
