#!/usr/bin/env python3
"""
Horizon MCP Server Startup Script

This script starts the Model Context Protocol server for Horizon AI Assistant.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MCP module
from mcp import create_mcp_server
from core import setup_logging, validate_config

def start_mcp_server():
    """Start the MCP server."""
    logger = setup_logging("HorizonMCPServer")
    
    logger.info("🤖 Starting Horizon MCP Server...")
    logger.info("🔗 MCP server will communicate via stdio")
    
    # Validate configuration
    config_status = validate_config()
    logger.info(f"📊 Configuration status: {config_status}")
    
    try:
        logger.info("✅ MCP server starting...")
        server = create_mcp_server()
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("🚪 MCP server stopped by user")
    except Exception as e:
        logger.error(f"❌ MCP server error: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Start Horizon MCP Server')
    parser.add_argument('--version', action='store_true', help='Show version and exit')
    
    args = parser.parse_args()
    
    if args.version:
        from mcp import __version__, __mcp_version__
        print(f"Horizon MCP Server v{__version__}")
        print(f"MCP Protocol Version: {__mcp_version__}")
        return
    
    start_mcp_server()

if __name__ == '__main__':
    main()
