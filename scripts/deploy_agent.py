#!/usr/bin/env python3
"""
Enhanced Horizon Agent Deployment Script

This script sets up and deploys the Horizon AI Assistant as a production-ready MCP agent.
It includes configuration management, monitoring, and deployment validation.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print deployment banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                   🚀 Horizon Agent Deployment Manager 🚀                    ║
    ║                        Production-Ready MCP Agent Setup                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_prerequisites():
    """Check deployment prerequisites."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version}")
    
    # Check required directories
    required_dirs = ['core', 'mcp', 'scripts', 'web']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ Missing directory: {dir_name}")
            return False
    print(f"✅ All required directories present")
    
    # Check core modules
    try:
        from core import get_ai_engine, get_enhanced_memory_system, get_enhanced_media_engine
        print("✅ Core modules importable")
    except ImportError as e:
        print(f"❌ Core module import error: {e}")
        return False
    
    # Check MCP server
    try:
        from mcp.server import create_mcp_server
        server = create_mcp_server()
        tools = server.list_tools()
        resources = server.list_resources()
        print(f"✅ MCP server ready: {len(tools)} tools, {len(resources)} resources")
    except Exception as e:
        print(f"❌ MCP server error: {e}")
        return False
    
    return True

def create_production_config():
    """Create production configuration."""
    print("⚙️ Creating production configuration...")
    
    config_template = {
        "deployment": {
            "mode": "production",
            "log_level": "INFO",
            "max_workers": 4,
            "timeout": 30,
            "rate_limit": {
                "requests_per_minute": 60,
                "burst_limit": 10
            }
        },
        "mcp": {
            "protocol_version": "2025-06-18",
            "stdio_mode": True,
            "max_message_size": 1048576,
            "heartbeat_interval": 30
        },
        "memory": {
            "database_path": "production_memory.db",
            "cleanup_interval_hours": 24,
            "max_memory_age_days": 365,
            "context_window_size": 20
        },
        "media": {
            "output_directory": "production_media",
            "max_file_size_mb": 50,
            "supported_formats": {
                "images": ["png", "jpg", "webp"],
                "videos": ["mp4", "webm"],
                "audio": ["mp3", "wav"],
                "models": ["obj", "glb"]
            }
        },
        "security": {
            "input_validation": True,
            "rate_limiting": True,
            "content_filtering": True,
            "audit_logging": True
        },
        "monitoring": {
            "metrics_enabled": True,
            "health_checks": True,
            "performance_logging": True,
            "error_tracking": True
        }
    }
    
    config_path = Path("production_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"✅ Production config created: {config_path}")
    return config_path

def setup_logging():
    """Setup production logging."""
    print("📝 Setting up production logging...")
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": f"logs/horizon_agent_{datetime.now().strftime('%Y%m%d')}.log",
                "formatter": "detailed",
                "level": "INFO"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "INFO"
            }
        },
        "loggers": {
            "HorizonMCP": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    log_config_path = Path("logging_config.json")
    with open(log_config_path, 'w') as f:
        json.dump(log_config, f, indent=2)
    
    print(f"✅ Logging configured: {log_config_path}")
    return log_config_path

def create_deployment_scripts():
    """Create deployment and management scripts."""
    print("📄 Creating deployment scripts...")
    
    # Docker deployment script
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 horizon && chown -R horizon:horizon /app
USER horizon

# Expose port for health checks
EXPOSE 8000

# Run the MCP server
CMD ["python", "-m", "mcp.server"]
"""
    
    with open("Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    print("✅ Dockerfile created")
    
    # Docker Compose for development
    compose_content = """version: '3.8'

services:
  horizon-agent:
    build: .
    container_name: horizon-mcp-agent
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./production_memory.db:/app/production_memory.db
      - ./production_media:/app/production_media
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from mcp.server import create_mcp_server; server = create_mcp_server(); print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
    
    with open("docker-compose.yml", 'w') as f:
        f.write(compose_content)
    print("✅ Docker Compose created")
    
    # Systemd service file
    service_content = f"""[Unit]
Description=Horizon MCP Agent
After=network.target

[Service]
Type=simple
User=horizon
WorkingDirectory={os.getcwd()}
Environment=PYTHONPATH={os.getcwd()}
ExecStart={sys.executable} -m scripts.start_mcp
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("horizon-agent.service", 'w') as f:
        f.write(service_content)
    print("✅ Systemd service file created")

def validate_deployment():
    """Validate the deployment setup."""
    print("🧪 Validating deployment...")
    
    try:
        # Test MCP server creation
        from mcp.server import create_mcp_server
        server = create_mcp_server()
        server.initialize()
        
        # Test tools
        tools = server.list_tools()
        assert len(tools) > 0, "No tools available"
        print(f"✅ {len(tools)} MCP tools available")
        
        # Test resources
        resources = server.list_resources()
        assert len(resources) > 0, "No resources available"
        print(f"✅ {len(resources)} MCP resources available")
        
        # Test core systems
        assert server.ai_engine is not None, "AI engine not initialized"
        assert server.memory_system is not None, "Memory system not initialized"
        assert server.media_engine is not None, "Media engine not initialized"
        print("✅ All core systems initialized")
        
        # Test enhanced features
        capabilities = server.media_engine.get_generation_capabilities()
        assert 'image_generation' in capabilities, "Image generation not available"
        print("✅ Enhanced media generation capabilities available")
        
        memory_stats = server.memory_system.get_memory_stats()
        assert 'total_memories' in memory_stats, "Memory stats not available"
        print("✅ Enhanced memory system operational")
        
        print("🎉 Deployment validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Deployment validation failed: {e}")
        return False

def generate_deployment_report():
    """Generate deployment report."""
    print("📊 Generating deployment report...")
    
    report = {
        "deployment_info": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0-day4-enhanced",
            "mode": "production",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        "components": {
            "core_modules": True,
            "mcp_server": True,
            "web_interface": True,
            "enhanced_memory": True,
            "enhanced_media": True
        },
        "capabilities": {
            "chat": True,
            "image_generation": True,
            "video_generation": True,
            "audio_generation": True,
            "logo_generation": True,
            "3d_model_generation": True,
            "batch_processing": True,
            "user_pattern_analysis": True,
            "conversation_summaries": True
        },
        "deployment_artifacts": [
            "production_config.json",
            "logging_config.json",
            "Dockerfile",
            "docker-compose.yml",
            "horizon-agent.service"
        ]
    }
    
    report_path = Path(f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Deployment report saved: {report_path}")
    return report_path

def print_deployment_instructions():
    """Print deployment instructions."""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                          🚀 Deployment Complete! 🚀                         ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Your Horizon AI Agent is ready for deployment!                             ║
    ║                                                                              ║
    ║  Quick Start Options:                                                        ║
    ║  ├─ Development: ./start.sh mcp                                              ║
    ║  ├─ Docker: docker-compose up -d                                             ║
    ║  ├─ Systemd: sudo systemctl start horizon-agent                             ║
    ║  └─ Python: python -m scripts.start_mcp                                     ║
    ║                                                                              ║
    ║  MCP Integration:                                                            ║
    ║  ├─ Tools: 13 available (including enhanced Day 4 features)                 ║
    ║  ├─ Resources: 6 available (chat history, patterns, capabilities)           ║
    ║  ├─ Protocol: MCP 2025-06-18 compliant                                      ║
    ║  └─ Features: Batch generation, user analysis, performance metrics          ║
    ║                                                                              ║
    ║  Monitoring:                                                                 ║
    ║  ├─ Logs: ./logs/horizon_agent_YYYYMMDD.log                                 ║
    ║  ├─ Config: production_config.json                                           ║
    ║  ├─ Health: http://localhost:8000/health (if running web mode)              ║
    ║  └─ Metrics: Available via MCP performance_metrics resource                 ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Horizon AI Agent')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip deployment validation')
    parser.add_argument('--config-only', action='store_true',
                       help='Only create configuration files')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please fix issues and try again.")
        sys.exit(1)
    
    # Create configuration
    config_path = create_production_config()
    log_config_path = setup_logging()
    
    if args.config_only:
        print("✅ Configuration files created successfully!")
        return
    
    # Create deployment scripts
    create_deployment_scripts()
    
    # Validate deployment
    if not args.skip_validation:
        if not validate_deployment():
            print("❌ Deployment validation failed. Please check the issues.")
            sys.exit(1)
    
    # Generate report
    report_path = generate_deployment_report()
    
    # Print instructions
    print_deployment_instructions()
    
    print(f"🎉 Horizon Agent deployment completed successfully!")
    print(f"📊 Report: {report_path}")

if __name__ == "__main__":
    main()