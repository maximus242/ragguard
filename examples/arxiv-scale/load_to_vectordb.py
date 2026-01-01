#!/usr/bin/env python3
"""
Load ArXiv papers into a vector database with embeddings.

Supports Qdrant and pgvector backends.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_papers(input_file: str) -> List[Dict]:
    """Load papers/chunks from JSON file."""
    with open(input_file, "r") as f:
        data = json.load(f)

    # Detect format: old (papers with abstracts) vs new (chunks)
    if data and "chunk_index" in data[0]:
        print(f"📁 Loaded {len(data)} chunks from {input_file}")
        # Count unique papers
        unique_papers = len(set(d.get("paper_id", d.get("id")) for d in data))
        avg_chunks = len(data) / unique_papers if unique_papers > 0 else 0
        print(f"   {unique_papers} papers, ~{avg_chunks:.1f} chunks per paper")
    else:
        print(f"📁 Loaded {len(data)} papers from {input_file}")

    return data


def load_to_qdrant(papers: List[Dict], collection_name: str, host: str, port: int):
    """Load papers into Qdrant vector database."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("   pip install qdrant-client sentence-transformers")
        return

    print(f"\n🚀 Loading papers into Qdrant...")
    print(f"   Host: {host}:{port}")
    print(f"   Collection: {collection_name}")

    # Connect to Qdrant
    client = QdrantClient(host=host, port=port)

    # Load embedding model
    print("\n📦 Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector_size = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors

    # Create collection
    print(f"\n🗂️  Creating collection '{collection_name}'...")
    try:
        client.delete_collection(collection_name)
    except:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # Process papers in batches
    batch_size = 100
    points = []

    # Detect format
    is_chunked = papers and "chunk_index" in papers[0]
    item_type = "chunks" if is_chunked else "papers"

    print(f"\n🔄 Processing {len(papers)} {item_type}...")
    for i, item in enumerate(tqdm(papers, desc=f"Embedding {item_type}")):
        # Create searchable text
        if is_chunked:
            # New format: chunks with 'text' field
            text = item.get("text", "")
        else:
            # Old format: papers with title + abstract
            text = f"{item['title']} {item['abstract']}"

        # Skip if text is empty or invalid
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            print(f"\n⚠️  Skipping chunk {i} (empty or invalid text)")
            continue

        # Generate embedding
        try:
            embedding = model.encode(text).tolist()
        except Exception as e:
            print(f"\n⚠️  Failed to encode chunk {i}: {e}")
            continue

        # Prepare metadata (payload)
        payload = {
            "id": item["id"],
            "title": item["title"],
            "text": item.get("text", item.get("abstract", ""))[:500],  # Truncate for storage
            "authors": item["authors"],
            "categories": item["categories"],
            "institution": item["institution"],
            "access_level": item["access_level"],
            "published": item["published"],
            "citation_count": item.get("citation_count", 0)
        }

        # Add chunk-specific fields if present
        if is_chunked:
            payload["paper_id"] = item.get("paper_id", item["id"].split("_")[0])
            payload["chunk_index"] = item.get("chunk_index", 0)
            payload["total_chunks"] = item.get("total_chunks", 1)
            payload["source"] = item.get("source", "unknown")

        # Create point
        point = PointStruct(
            id=i,
            vector=embedding,
            payload=payload
        )
        points.append(point)

        # Upload batch
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []

    # Upload remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"\n✅ Successfully loaded {len(papers)} {item_type} into Qdrant")
    print(f"   Collection: {collection_name}")
    print(f"   Vector size: {vector_size}D")
    print(f"   Distance metric: Cosine")
    if is_chunked:
        unique_papers = len(set(p.get("paper_id", p["id"].split("_")[0]) for p in papers))
        print(f"   Unique papers: {unique_papers}")
        print(f"   Avg chunks/paper: {len(papers) / unique_papers:.1f}")


def load_to_pgvector(papers: List[Dict], table_name: str, connection_string: str):
    """Load papers into PostgreSQL with pgvector."""
    try:
        import psycopg2
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("   pip install psycopg2-binary sentence-transformers")
        return

    print(f"\n🚀 Loading papers into pgvector...")
    print(f"   Connection: {connection_string.split('@')[1] if '@' in connection_string else connection_string}")
    print(f"   Table: {table_name}")

    # Connect to PostgreSQL
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Detect format
    is_chunked = papers and "chunk_index" in papers[0]

    # Create table
    print(f"\n🗂️  Creating table '{table_name}'...")
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    if is_chunked:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT,
                chunk_index INTEGER,
                total_chunks INTEGER,
                source TEXT,
                authors JSONB,
                categories JSONB,
                institution TEXT,
                access_level TEXT,
                published DATE,
                citation_count INTEGER,
                embedding vector(384)
            )
        """)
    else:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                paper_id TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                authors JSONB,
                categories JSONB,
                institution TEXT,
                access_level TEXT,
                published DATE,
                citation_count INTEGER,
                embedding vector(384)
            )
        """)

    # Create indexes for filtering
    cur.execute(f"CREATE INDEX ON {table_name} (institution)")
    cur.execute(f"CREATE INDEX ON {table_name} (access_level)")
    cur.execute(f"CREATE INDEX ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    conn.commit()

    # Load embedding model
    print("\n📦 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Insert papers/chunks
    item_type = "chunks" if is_chunked else "papers"
    print(f"\n🔄 Processing {len(papers)} {item_type}...")

    for item in tqdm(papers, desc=f"Inserting {item_type}"):
        # Create searchable text
        if is_chunked:
            text = item["text"]
        else:
            text = f"{item['title']} {item['abstract']}"

        embedding = model.encode(text).tolist()

        # Insert into database
        if is_chunked:
            cur.execute(f"""
                INSERT INTO {table_name}
                (chunk_id, paper_id, title, text, chunk_index, total_chunks, source,
                 authors, categories, institution, access_level, published, citation_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                item["id"],
                item.get("paper_id", item["id"].split("_")[0]),
                item["title"],
                item["text"][:2000],
                item.get("chunk_index", 0),
                item.get("total_chunks", 1),
                item.get("source", "unknown"),
                json.dumps(item["authors"]),
                json.dumps(item["categories"]),
                item["institution"],
                item["access_level"],
                item["published"],
                item.get("citation_count", 0),
                embedding
            ))
        else:
            cur.execute(f"""
                INSERT INTO {table_name}
                (paper_id, title, abstract, authors, categories, institution,
                 access_level, published, citation_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                item["id"],
                item["title"],
                item["abstract"][:1000],
                json.dumps(item["authors"]),
                json.dumps(item["categories"]),
                item["institution"],
                item["access_level"],
                item["published"],
                item.get("citation_count", 0),
                embedding
            ))

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✅ Successfully loaded {len(papers)} {item_type} into pgvector")
    if is_chunked:
        unique_papers = len(set(p.get("paper_id", p["id"].split("_")[0]) for p in papers))
        print(f"   Unique papers: {unique_papers}")
        print(f"   Avg chunks/paper: {len(papers) / unique_papers:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Load ArXiv papers into vector database"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/arxiv_papers.json",
        help="Input JSON file with papers"
    )
    parser.add_argument(
        "--db",
        type=str,
        choices=["qdrant", "pgvector"],
        default="qdrant",
        help="Vector database to use"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Database host (for Qdrant)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Database port (for Qdrant)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="arxiv_papers",
        help="Collection/table name"
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        default="postgresql://localhost/arxiv_rag",
        help="PostgreSQL connection string (for pgvector)"
    )

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  ArXiv Vector Database Loader                                ║
║                                                              ║
║  This will embed papers and load them into your vector DB    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Load papers
    papers = load_papers(args.input)

    # Load into database
    if args.db == "qdrant":
        load_to_qdrant(
            papers,
            collection_name=args.collection,
            host=args.host,
            port=args.port
        )
    elif args.db == "pgvector":
        load_to_pgvector(
            papers,
            table_name=args.collection,
            connection_string=args.connection_string
        )

    print(f"\n✨ Next steps:")
    print(f"   1. Run the demo: python app.py")
    print(f"   2. Run benchmarks: python benchmark.py")
    print(f"   3. Interactive mode: python interactive_demo.py")


if __name__ == "__main__":
    main()
