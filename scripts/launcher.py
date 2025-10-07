#!/usr/bin/env python3
"""
Horizon AI Assistant Launcher

This script provides a unified entry point for starting Horizon in different modes.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import setup_logging

def show_banner():
    """Show the Horizon banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                 🌟 HORIZON AI ASSISTANT 🌟                ║
║               Advanced Multi-Modal AI System             ║
╠═══════════════════════════════════════════════════════════╣
║  🎯 Multi-Personality AI Conversations                   ║
║  🎨 Advanced Media Generation (Images, Videos, Audio)    ║
║  🧠 Intelligent Memory & Learning System                 ║
║  🌐 Web Interface & MCP Protocol Support                 ║
║  📊 Real-time Analytics & Insights                       ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main launcher entry point."""
    parser = argparse.ArgumentParser(
        description='Horizon AI Assistant Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  web     - Start web interface only (Flask server)
  mcp     - Start MCP server only (Model Context Protocol)
  both    - Start both web and MCP servers
  setup   - Set up development environment
  
Examples:
  python scripts/launcher.py web --port 8080
  python scripts/launcher.py mcp
  python scripts/launcher.py both --debug
  python scripts/launcher.py setup
        """
    )
    
    parser.add_argument('mode', choices=['web', 'mcp', 'both', 'setup'],
                       help='Launch mode')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host for web server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for web server (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-banner', action='store_true',
                       help='Skip banner display')
    parser.add_argument('--version', action='version', version='Horizon AI v1.0.0')
    
    args = parser.parse_args()
    
    if not args.no_banner:
        show_banner()
    
    logger = setup_logging("HorizonLauncher")
    
    try:
        if args.mode == 'web':
            logger.info("🌐 Starting Horizon in Web Mode...")
            from scripts.start_web import start_web_server
            start_web_server(args.host, args.port, args.debug)
            
        elif args.mode == 'mcp':
            logger.info("🤖 Starting Horizon in MCP Mode...")
            from scripts.start_mcp import start_mcp_server
            start_mcp_server()
            
        elif args.mode == 'both':
            logger.info("🚀 Starting Horizon in Dual Mode...")
            from scripts.start_both import start_both_servers
            start_both_servers(args.host, args.port, args.debug)
            
        elif args.mode == 'setup':
            logger.info("🛠️ Setting up Horizon development environment...")
            from scripts.dev_setup import setup_development_environment
            setup_development_environment()
            
    except KeyboardInterrupt:
        logger.info("👋 Horizon launcher stopped by user")
    except Exception as e:
        logger.error(f"❌ Launcher error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()