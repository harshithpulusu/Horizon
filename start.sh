#!/bin/bash

# Horizon AI Assistant Startup Script
echo "🌟 Starting Horizon AI Assistant..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p static/generated_images
mkdir -p static/generated_videos
mkdir -p static/generated_audio
mkdir -p static/generated_music
mkdir -p static/generated_3d_models
mkdir -p static/generated_avatars
mkdir -p static/generated_logos
mkdir -p static/generated_designs
mkdir -p static/generated_gifs
mkdir -p templates
mkdir -p logs
mkdir -p backups

# Determine launch mode based on arguments
MODE=${1:-web}
HOST=${2:-0.0.0.0}
PORT=${3:-5000}

echo "✅ Environment ready!"

case $MODE in
    "web")
        echo "🚀 Launching Horizon Web Interface..."
        echo "📱 Open your browser and go to: http://$HOST:$PORT"
        python scripts/launcher.py web --host $HOST --port $PORT
        ;;
    "mcp")
        echo "🚀 Launching Horizon MCP Server..."
        echo "🤖 MCP server will communicate via stdio"
        python scripts/launcher.py mcp
        ;;
    "both")
        echo "🚀 Launching Horizon Dual Mode..."
        echo "🌐 Web interface: http://$HOST:$PORT"
        echo "🤖 MCP server: stdio communication"
        python scripts/launcher.py both --host $HOST --port $PORT
        ;;
    "setup")
        echo "🚀 Setting up Horizon Development Environment..."
        python scripts/launcher.py setup
        ;;
    *)
        echo "Usage: $0 [web|mcp|both|setup] [host] [port]"
        echo "  web  - Start web interface (default)"
        echo "  mcp  - Start MCP server"
        echo "  both - Start both servers"
        echo "  setup - Setup development environment"
        echo ""
        echo "Examples:"
        echo "  $0 web 0.0.0.0 8080"
        echo "  $0 mcp"
        echo "  $0 both"
        echo "  $0 setup"
        exit 1
        ;;
esac
