"""
Horizon Web Interface Module

This module contains the Flask web application for Horizon AI Assistant.
It provides a user-friendly web interface for interacting with Horizon's
AI capabilities.

This is the refactored web interface that uses the shared core module
for business logic while maintaining the original user experience.

Modules:
- app: Main Flask application (refactored from original app.py)
- routes: Web route definitions
- static: CSS, JavaScript, and asset files
- templates: HTML template files
"""

__version__ = "1.0.0"
__author__ = "Horizon AI Team"

# Web module imports (to be created)
try:
    from .app import create_app, app
    from .routes import register_routes
    
    __all__ = [
        'create_app',
        'app', 
        'register_routes'
    ]
except ImportError:
    # Modules not yet created
    __all__ = []