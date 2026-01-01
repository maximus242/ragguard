#!/bin/bash
# Quick start script for ArXiv Scale demo

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  RAGGuard ArXiv Scale Demo - Quick Start                     ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker and docker-compose."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8+."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -q -r requirements.txt

# Start Qdrant
echo ""
echo "🚀 Starting Qdrant..."
docker-compose up -d qdrant

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to start..."
sleep 5

# Check if data already exists
if [ -f "data/arxiv_papers.json" ]; then
    echo "✅ Found existing data at data/arxiv_papers.json"
    read -p "   Download new papers? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📥 Downloading 10,000 ArXiv papers..."
        python3 download_arxiv.py --limit 10000
    fi
else
    echo "📥 Downloading 10,000 ArXiv papers (this may take 10-15 minutes)..."
    python3 download_arxiv.py --limit 10000
fi

# Load into Qdrant
echo ""
echo "🔄 Loading papers into Qdrant (this may take 5-10 minutes)..."
python3 load_to_vectordb.py --db qdrant

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎉 You can now:"
echo "   1. Run the benchmark:      python3 benchmark.py"
echo "   2. Interactive demo:       python3 interactive_demo.py"
echo "   3. Stop Qdrant:           docker-compose down"
echo ""
