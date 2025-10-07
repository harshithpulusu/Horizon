#!/usr/bin/env python3
"""
Horizon Dual Server Startup Script

This script starts both the web server and MCP server for Horizon AI Assistant.
"""

import os
import sys
import time
import signal
import argparse
import multiprocessing
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import setup_logging

# Global process references
web_process = None
mcp_process = None

def start_web_process(host='0.0.0.0', port=5000, debug=False):
    """Start web server in a separate process."""
    from scripts.start_web import start_web_server
    start_web_server(host, port, debug)

def start_mcp_process():
    """Start MCP server in a separate process."""
    from scripts.start_mcp import start_mcp_server
    start_mcp_server()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger = setup_logging("HorizonDualServer")
    logger.info("🚪 Shutting down both servers...")
    
    global web_process, mcp_process
    
    if web_process and web_process.is_alive():
        logger.info("🚪 Stopping web server...")
        web_process.terminate()
        web_process.join(timeout=5)
        if web_process.is_alive():
            web_process.kill()
    
    if mcp_process and mcp_process.is_alive():
        logger.info("🚪 Stopping MCP server...")
        mcp_process.terminate()
        mcp_process.join(timeout=5)
        if mcp_process.is_alive():
            mcp_process.kill()
    
    logger.info("✅ Both servers stopped")
    sys.exit(0)

def start_both_servers(host='0.0.0.0', port=5000, debug=False):
    """Start both web and MCP servers."""
    logger = setup_logging("HorizonDualServer")
    
    logger.info("🚀 Starting Horizon Dual Server Mode...")
    logger.info(f"🌐 Web server: http://{host}:{port}")
    logger.info("🤖 MCP server: stdio communication")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global web_process, mcp_process
    
    try:
        # Start web server process
        logger.info("🌐 Starting web server process...")
        web_process = multiprocessing.Process(
            target=start_web_process,
            args=(host, port, debug),
            name="HorizonWebServer"
        )
        web_process.start()
        
        # Start MCP server process  
        logger.info("🤖 Starting MCP server process...")
        mcp_process = multiprocessing.Process(
            target=start_mcp_process,
            name="HorizonMCPServer"
        )
        mcp_process.start()
        
        logger.info("✅ Both servers started successfully")
        logger.info("🔄 Press Ctrl+C to stop both servers")
        
        # Monitor processes
        while True:
            if not web_process.is_alive():
                logger.error("❌ Web server process died")
                break
            if not mcp_process.is_alive():
                logger.error("❌ MCP server process died")
                break
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"❌ Error running dual servers: {e}")
    finally:
        signal_handler(None, None)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Start both Horizon servers')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    start_both_servers(args.host, args.port, args.debug)

if __name__ == '__main__':
    main()
