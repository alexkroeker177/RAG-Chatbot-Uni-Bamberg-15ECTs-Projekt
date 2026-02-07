#!/bin/bash
# Setup script for RAG Examination Chatbot

set -e

echo "=== RAG Examination Chatbot Setup ==="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is required but not found."
    echo "Please install Python 3.12 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3.12 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION ✓"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3.12 -m venv venv
echo "Virtual environment created ✓"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated ✓"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "pip upgraded ✓"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed ✓"
echo ""

# Create config if it doesn't exist
if [ ! -f config.yaml ]; then
    echo "Creating config.yaml from example..."
    cp config.example.yaml config.yaml
    echo "config.yaml created ✓"
    echo "Please edit config.yaml with your settings."
else
    echo "config.yaml already exists, skipping..."
fi
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml with your Ollama server settings"
echo "  2. Place PDF files in Studienordnungen/ directory"
echo "  3. Run ingestion: python v1/ingest.py"
echo "  4. Start chatting: python v1/chat.py"
