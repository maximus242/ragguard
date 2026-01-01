#!/usr/bin/env bash
#
# Quickstart script to download 250K ArXiv papers with full PDFs
# for the viral HN benchmark.
#
# This will:
# - Download 250K research papers from ArXiv
# - Extract full text from PDFs
# - Chunk into ~23 million text segments
# - Require ~350GB storage
# - Take ~24-48 hours (ArXiv rate limits)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PAPER_LIMIT=${1:-250000}
OUTPUT_DIR=${2:-"./data"}
BATCH_SIZE=100
CHUNK_SIZE=1000

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  RAGGuard Viral HN Demo: Download 250K Papers                ║"
echo "║                                                              ║"
echo "║  This will create the MOST COMPELLING benchmark for HN      ║"
echo "║  by testing RAGGuard at production scale with real PDFs.    ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check disk space
echo -e "${YELLOW}📊 Checking disk space...${NC}"
AVAILABLE_GB=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_GB=400  # 350GB + buffer

echo "   Available: ${AVAILABLE_GB}GB"
echo "   Required:  ${REQUIRED_GB}GB"

if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo -e "${RED}❌ Insufficient disk space!${NC}"
    echo "   Need at least ${REQUIRED_GB}GB"
    exit 1
fi

echo -e "${GREEN}✅ Sufficient disk space${NC}"
echo

# Show what will happen
echo -e "${CYAN}📋 Download Plan:${NC}"
echo "   Papers:     $PAPER_LIMIT"
echo "   Mode:       Full PDFs with chunking"
echo "   Batch size: $BATCH_SIZE papers"
echo "   Chunk size: $CHUNK_SIZE characters"
echo "   Output:     $OUTPUT_DIR/arxiv_papers_250k.json"
echo
echo -e "${CYAN}📊 Estimates:${NC}"
echo "   Chunks:     ~23 million (92 chunks/paper)"
echo "   Storage:    ~350GB final"
echo "   Time:       24-48 hours (ArXiv rate limits)"
echo
echo -e "${YELLOW}⚠️  This will:${NC}"
echo "   - Run for 24-48 hours (do NOT close terminal)"
echo "   - Download ~250K PDFs temporarily"
echo "   - Extract and chunk all text"
echo "   - Save progress after each batch"
echo "   - Auto-cleanup PDFs after processing"
echo

# Confirm
read -p "$(echo -e ${GREEN}Continue? [yes/no]:${NC} )" -r
echo

if [[ ! $REPLY =~ ^[Yy](es)?$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"

if ! python3 -c "import PyPDF2" 2>/dev/null; then
    echo -e "${YELLOW}Installing PyPDF2...${NC}"
    pip install PyPDF2
fi

if ! python3 -c "import rich" 2>/dev/null; then
    echo -e "${YELLOW}Installing rich...${NC}"
    pip install rich
fi

echo -e "${GREEN}✅ Dependencies ready${NC}"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save start time
START_TIME=$(date +%s)
START_DATE=$(date)

echo -e "${GREEN}🚀 Starting download...${NC}"
echo "   Started: $START_DATE"
echo "   Output:  $OUTPUT_DIR/arxiv_papers_250k.json"
echo "   Log:     $OUTPUT_DIR/download.log"
echo

# Run download (with logging)
python3 download_arxiv.py \
    --limit "$PAPER_LIMIT" \
    --pdfs \
    --batch-size "$BATCH_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    --output "$OUTPUT_DIR/arxiv_papers_250k.json" \
    2>&1 | tee "$OUTPUT_DIR/download.log"

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo
echo -e "${GREEN}✅ Download complete!${NC}"
echo "   Duration: ${HOURS}h ${MINUTES}m"
echo "   Output:   $OUTPUT_DIR/arxiv_papers_250k.json"
echo

# Show file size
FILE_SIZE=$(du -h "$OUTPUT_DIR/arxiv_papers_250k.json" | cut -f1)
echo "   File size: $FILE_SIZE"

# Next steps
echo
echo -e "${CYAN}✨ Next steps:${NC}"
echo "   1. Load into Qdrant:"
echo "      python load_to_vectordb.py --input $OUTPUT_DIR/arxiv_papers_250k.json --db qdrant"
echo
echo "   2. Run multi-scale benchmark:"
echo "      python benchmark_multiscale.py --input $OUTPUT_DIR/arxiv_papers_250k.json"
echo
echo "   3. Create HN post with results"
echo

echo -e "${GREEN}🎉 Ready for viral HN post!${NC}"
