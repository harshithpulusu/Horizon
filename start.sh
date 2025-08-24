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
mkdir -p static
mkdir -p templates

# Check if all files exist
required_files=("app.py" "templates/index.html" "static/enhanced_ai.js")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All dependencies installed and files verified!"
echo "🚀 Starting Horizon AI Assistant..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🎤 Make sure to allow microphone access for voice features!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

# Start the Flask application
python app.py
