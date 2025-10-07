"""
Horizon Startup Scripts

This directory contains scripts to start different components of Horizon:

- start_web.py: Start the web interface only
- start_mcp.py: Start the MCP agent only  
- start_both.py: Start both web and MCP services
- dev_setup.py: Development environment setup
"""

__version__ = "1.0.0"
__author__ = "Horizon AI Team"

# Startup scripts imports
try:
    from .start_web import start_web_server
    from .start_mcp import start_mcp_server
    from .start_both import start_both_servers
    from .dev_setup import setup_development_environment
    
    __all__ = [
        'start_web_server',
        'start_mcp_server', 
        'start_both_servers',
        'setup_development_environment'
    ]
except ImportError:
    # Scripts not yet created
    __all__ = []