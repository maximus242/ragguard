#!/usr/bin/env python3
"""
Download ArXiv papers for the RAGGuard scale demo.

Supports two modes:
1. Metadata only (fast, ~500 bytes per paper)
2. Full PDFs with chunking (realistic RAG, ~1MB per paper)

Uses batch processing to keep storage manageable.
"""

import argparse
import json
import random
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# Simulated institutions (for demo purposes)
INSTITUTIONS = [
    "MIT", "Stanford", "Berkeley", "CMU", "Harvard",
    "Oxford", "Cambridge", "ETH Zurich", "Tokyo", "Toronto",
    "Princeton", "Yale", "Columbia", "Cornell", "UPenn"
]

# Access levels
ACCESS_LEVELS = ["public", "institution", "restricted"]


def fetch_arxiv_batch(category: str, start: int, max_results: int = 1000) -> List[Dict]:
    """
    Fetch a batch of papers from ArXiv API.

    Args:
        category: ArXiv category (e.g., "cs.AI", "physics.quantum")
        start: Start index for pagination
        max_results: Number of results per request

    Returns:
        List of paper metadata dicts
    """
    base_url = "http://export.arxiv.org/api/query?"
    query = f"cat:{category}"

    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    url = base_url + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()

        # Parse XML response
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            # Extract paper data
            paper_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

            # Authors
            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
            ]

            # Categories
            categories = [
                cat.attrib["term"]
                for cat in entry.findall("atom:category", ns)
            ]

            # Published date
            published = entry.find("atom:published", ns).text[:10]

            # Simulate institution and access level (for demo)
            institution = random.choice(INSTITUTIONS)
            access_level = random.choices(
                ACCESS_LEVELS,
                weights=[0.6, 0.3, 0.1]  # 60% public, 30% institution, 10% restricted
            )[0]

            # Get PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "categories": categories,
                "published": published,
                "institution": institution,
                "access_level": access_level,
                "citation_count": random.randint(0, 10000),  # Simulated
                "pdf_url": pdf_url
            })

        return papers

    except Exception as e:
        print(f"❌ Error fetching batch (start={start}): {e}")
        return []


def download_pdf(paper: Dict, pdf_dir: Path) -> Optional[Path]:
    """
    Download a single PDF file.

    Args:
        paper: Paper metadata dict with pdf_url
        pdf_dir: Directory to save PDF

    Returns:
        Path to downloaded PDF, or None if download failed
    """
    if not paper.get("pdf_url"):
        return None

    pdf_path = pdf_dir / f"{paper['id']}.pdf"

    # Skip if already downloaded
    if pdf_path.exists():
        return pdf_path

    try:
        # Download PDF
        urllib.request.urlretrieve(paper["pdf_url"], pdf_path)
        return pdf_path
    except Exception as e:
        print(f"   ⚠️  Failed to download {paper['id']}: {e}")
        return None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from PDF using PyPDF2.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text, or None if extraction failed
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        return text.strip()
    except ImportError:
        print("⚠️  PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"   ⚠️  Failed to extract text from {pdf_path.name}: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size // 2:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - chunk_overlap

    return [c for c in chunks if c]  # Filter empty chunks


def process_pdf_batch(
    papers: List[Dict],
    pdf_dir: Path,
    output_chunks: List[Dict],
    download_pdfs: bool = True,
    chunk_size: int = 1000
) -> int:
    """
    Process a batch of papers: download PDFs, extract text, chunk, and delete PDFs.

    Args:
        papers: List of paper metadata dicts
        pdf_dir: Directory for temporary PDF storage
        output_chunks: List to append chunks to
        download_pdfs: Whether to download PDFs (False = metadata only)
        chunk_size: Size of text chunks

    Returns:
        Number of chunks created
    """
    chunks_created = 0

    if not download_pdfs:
        # Metadata only mode - just use abstracts
        for paper in papers:
            chunk = {
                "id": f"{paper['id']}_0",
                "paper_id": paper["id"],
                "title": paper["title"],
                "text": paper["abstract"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "published": paper["published"],
                "institution": paper["institution"],
                "access_level": paper["access_level"],
                "citation_count": paper["citation_count"],
                "chunk_index": 0,
                "total_chunks": 1
            }
            output_chunks.append(chunk)
            chunks_created += 1
        return chunks_created

    # Full PDF mode with chunking
    for i, paper in enumerate(papers):
        # Download PDF
        pdf_path = download_pdf(paper, pdf_dir)
        if not pdf_path:
            # Fallback to abstract if PDF download fails
            chunk = {
                "id": f"{paper['id']}_0",
                "paper_id": paper["id"],
                "title": paper["title"],
                "text": paper["abstract"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "published": paper["published"],
                "institution": paper["institution"],
                "access_level": paper["access_level"],
                "citation_count": paper["citation_count"],
                "chunk_index": 0,
                "total_chunks": 1,
                "source": "abstract"
            }
            output_chunks.append(chunk)
            chunks_created += 1
            continue

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            text = paper["abstract"]  # Fallback

        # Chunk text
        text_chunks = chunk_text(text, chunk_size=chunk_size)

        # Create chunk metadata
        for j, chunk_content in enumerate(text_chunks):
            chunk = {
                "id": f"{paper['id']}_{j}",
                "paper_id": paper["id"],
                "title": paper["title"],
                "text": chunk_content,
                "authors": paper["authors"],
                "categories": paper["categories"],
                "published": paper["published"],
                "institution": paper["institution"],
                "access_level": paper["access_level"],
                "citation_count": paper["citation_count"],
                "chunk_index": j,
                "total_chunks": len(text_chunks),
                "source": "pdf"
            }
            output_chunks.append(chunk)
            chunks_created += 1

        # Delete PDF to save space
        pdf_path.unlink()

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(papers)} PDFs ({chunks_created} chunks)")

    return chunks_created


def download_papers(
    limit: int,
    output_file: str,
    categories: List[str],
    download_pdfs: bool = False,
    batch_size: int = 100,
    chunk_size: int = 1000
):
    """
    Download papers from multiple ArXiv categories with optional PDF processing.

    Args:
        limit: Total number of papers to download
        output_file: Output JSON file path
        categories: List of ArXiv categories to fetch from
        download_pdfs: Whether to download full PDFs (vs metadata only)
        batch_size: Number of papers to process before saving (for PDF mode)
        chunk_size: Size of text chunks for PDF mode
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_dir = output_path.parent / "pdfs_temp"
    if download_pdfs:
        pdf_dir.mkdir(exist_ok=True)

    all_chunks = []
    papers_per_category = max(1, limit // len(categories))
    total_downloaded = 0
    total_chunks = 0

    mode = "PDF mode (with chunking)" if download_pdfs else "metadata only"
    print(f"📥 Downloading {limit} papers from ArXiv ({mode})...")
    print(f"   Categories: {', '.join(categories)}")
    print(f"   ~{papers_per_category} papers per category")
    if download_pdfs:
        print(f"   Batch size: {batch_size} papers")
        print(f"   Chunk size: {chunk_size} characters")
    print()

    for category in categories:
        print(f"📚 Fetching from {category}...")

        category_papers = []
        start = 0
        api_batch_size = min(100, papers_per_category)

        while len(category_papers) < papers_per_category:
            # Fetch metadata batch
            papers = fetch_arxiv_batch(category, start, api_batch_size)

            if not papers:
                print(f"   ⚠️  No more papers available for {category}")
                break

            category_papers.extend(papers)
            start += api_batch_size

            print(f"   ✓ Fetched metadata {len(category_papers)}/{papers_per_category}")

            # Be nice to ArXiv API (rate limit: 3 seconds between requests)
            time.sleep(3)

            # Process in batches if downloading PDFs
            if download_pdfs and len(category_papers) >= batch_size:
                batch = category_papers[:batch_size]
                category_papers = category_papers[batch_size:]

                print(f"   📄 Processing batch of {len(batch)} PDFs...")
                chunks = process_pdf_batch(batch, pdf_dir, all_chunks, download_pdfs, chunk_size)
                total_chunks += chunks
                total_downloaded += len(batch)

                # Save progress
                with open(output_path, "w") as f:
                    json.dump(all_chunks, f, indent=2)

                print(f"   ✅ Saved {total_chunks} chunks from {total_downloaded} papers")

            if len(category_papers) >= papers_per_category:
                break

        # Process remaining papers in category
        if category_papers:
            remaining = category_papers[:papers_per_category - total_downloaded % papers_per_category]
            print(f"   📄 Processing final batch of {len(remaining)} papers...")
            chunks = process_pdf_batch(remaining, pdf_dir, all_chunks, download_pdfs, chunk_size)
            total_chunks += chunks
            total_downloaded += len(remaining)

        print(f"   ✅ Completed {category}: {total_downloaded} papers, {total_chunks} chunks\n")

    # Final save
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    # Clean up temp PDF directory
    if download_pdfs and pdf_dir.exists():
        shutil.rmtree(pdf_dir)

    # Print statistics
    print(f"\n✅ Processing complete!")
    print(f"   Papers processed: {total_downloaded}")
    print(f"   Total chunks: {total_chunks}")
    if download_pdfs and total_downloaded > 0:
        print(f"   Avg chunks per paper: {total_chunks / total_downloaded:.1f}")
    print(f"📁 Saved to: {output_file}")

    print("\n📊 Statistics:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Institutions: {len(set(c['institution'] for c in all_chunks))}")
    print(f"   Categories: {len(set(cat for c in all_chunks for cat in c['categories']))}")

    access_counts = {}
    source_counts = {}
    for chunk in all_chunks:
        access_counts[chunk["access_level"]] = access_counts.get(chunk["access_level"], 0) + 1
        source = chunk.get("source", "abstract")
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n   Access Levels:")
    for level, count in sorted(access_counts.items()):
        pct = (count / len(all_chunks)) * 100
        print(f"     {level}: {count} ({pct:.1f}%)")

    if download_pdfs:
        print(f"\n   Sources:")
        for source, count in sorted(source_counts.items()):
            pct = (count / len(all_chunks)) * 100
            print(f"     {source}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Download ArXiv papers for RAGGuard scale demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10K papers (metadata only, fast)
  python download_arxiv.py --limit 10000

  # Download 250K papers with full PDFs (for viral HN demo)
  python download_arxiv.py --limit 250000 --pdfs --batch-size 100

  # Quick test with 100 papers and PDFs
  python download_arxiv.py --limit 100 --pdfs

Storage estimates:
  Metadata only:  ~500 bytes/paper  (10K = ~5MB, 250K = ~125MB)
  With PDFs:      ~1.5GB/1K chunks  (250K papers = ~2M chunks = ~300GB)
        """
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of papers to download (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/arxiv_papers.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.IR"],
        help="ArXiv categories to download from"
    )
    parser.add_argument(
        "--pdfs",
        action="store_true",
        help="Download full PDFs and extract/chunk text (slower, requires more storage)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of papers to process per batch (for PDF mode, default: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks in characters (for PDF mode, default: 1000)"
    )

    args = parser.parse_args()

    # Estimate storage
    if args.pdfs:
        # Assume ~8 chunks per paper, ~1.5KB per chunk
        chunks_estimate = args.limit * 8
        storage_mb = chunks_estimate * 1.5 / 1000
        storage_gb = storage_mb / 1000
        mode = "PDF mode (with chunking)"
        storage_note = f"~{storage_gb:.1f}GB final storage"
    else:
        storage_mb = args.limit * 0.5 / 1000
        mode = "metadata only"
        storage_note = f"~{storage_mb:.1f}MB storage"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ArXiv Paper Downloader for RAGGuard Scale Demo             ║
║                                                              ║
║  Mode: {mode:40s}              ║
║  Papers: {args.limit:10,d} ({storage_note:30s})             ║
║                                                              ║
║  This will download papers from ArXiv.org                    ║
║  Please be respectful of ArXiv's API rate limits.           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    if args.pdfs and args.limit > 10000:
        import sys
        # Only prompt if running interactively
        if sys.stdin.isatty():
            print("⚠️  WARNING: Downloading 10K+ papers with PDFs will:")
            print("   - Take several hours (ArXiv rate limits)")
            print(f"   - Require ~{storage_gb:.1f}GB of storage")
            print("   - Put load on ArXiv's servers")
            print()
            response = input("Continue? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Cancelled.")
                return
        else:
            print("⚠️  Non-interactive mode: Starting large download automatically...")
            print(f"   Downloading {args.limit:,} papers with PDFs")
            print(f"   Estimated storage: ~{storage_gb:.1f}GB")
            print()

    start_time = time.time()

    download_papers(
        limit=args.limit,
        output_file=args.output,
        categories=args.categories,
        download_pdfs=args.pdfs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )

    elapsed = time.time() - start_time
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60

    print(f"\n⏱️  Total time: {int(hours)}h {int(minutes)}m")
    print(f"\n✨ Next step: Load chunks into vector database")
    print(f"   python load_to_vectordb.py --input {args.output}")


if __name__ == "__main__":
    main()
