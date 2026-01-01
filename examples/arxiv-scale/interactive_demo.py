#!/usr/bin/env python3
"""
Interactive demo for querying ArXiv papers with RAGGuard.
"""

import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel


console = Console()


USERS = [
    {
        "name": "Alice Chen",
        "institution": "MIT",
        "department": "Computer Science",
        "roles": ["researcher"],
        "id": "alice@mit.edu"
    },
    {
        "name": "Bob Smith",
        "institution": "Stanford",
        "department": "Physics",
        "roles": ["researcher"],
        "id": "bob@stanford.edu"
    },
    {
        "name": "Charlie Davis",
        "institution": None,
        "department": None,
        "roles": ["public"],
        "id": "charlie@email.com"
    },
    {
        "name": "Diana Admin",
        "institution": "MIT",
        "department": "Administration",
        "roles": ["admin"],
        "id": "diana@mit.edu"
    }
]


def select_user() -> dict:
    """Prompt user to select their identity."""
    console.print("\n[bold cyan]👤 Select Your Identity:[/bold cyan]\n")

    table = Table(show_header=True)
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Name", style="green")
    table.add_column("Institution")
    table.add_column("Role")

    for i, user in enumerate(USERS, 1):
        institution = user.get("institution", "Public")
        role = ", ".join(user["roles"])
        table.add_row(str(i), user["name"], institution, role)

    console.print(table)
    console.print()

    choice = IntPrompt.ask("Select user", choices=[str(i) for i in range(1, len(USERS) + 1)])
    return USERS[choice - 1]


def search_papers(retriever, query: str, user: dict, limit: int = 10):
    """Search papers and display results."""
    import time

    console.print(f"\n🔍 Searching for: [yellow]{query}[/yellow]")
    console.print(f"👤 User: {user['name']} ({user.get('institution', 'Public')})")
    console.print()

    start = time.time()
    results = retriever.search(
        query=query,
        user=user,
        limit=limit
    )
    latency = (time.time() - start) * 1000

    if not results:
        console.print("[red]❌ No accessible papers found[/red]")
        return

    console.print(f"✅ Found [green]{len(results)}[/green] accessible papers ([cyan]{latency:.1f}ms[/cyan])\n")

    for i, result in enumerate(results, 1):
        # Extract metadata from ScoredPoint object
        if hasattr(result, 'payload'):
            # Qdrant ScoredPoint
            payload = result.payload
            title = payload.get("title", "Unknown")
            institution = payload.get("institution", "Unknown")
            access_level = payload.get("access_level", "unknown")
            categories = payload.get("categories", [])
            score = result.score if hasattr(result, 'score') else 0
            citations = payload.get("citation_count", 0)
        else:
            # Dict format (fallback)
            title = result.get("title", "Unknown")
            institution = result.get("institution", "Unknown")
            access_level = result.get("access_level", "unknown")
            categories = result.get("categories", [])
            score = result.get("score", 0)
            citations = result.get("citation_count", 0)

        # Format categories
        cat_str = categories[0] if categories else "N/A"

        # Access indicator
        if access_level == "public":
            access_icon = "⭐"
        elif access_level == "institution":
            access_icon = "🔒"
        else:
            access_icon = "🚫"

        # Print result
        console.print(f"{i}. {access_icon} [bold]{title}[/bold]")
        console.print(f"   [{cat_str}] {institution} • {citations:,} citations • {score:.3f} relevance")
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Interactive ArXiv search demo")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="arxiv_papers", help="Collection name")

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]🔬 ArXiv Research Assistant[/bold cyan]\n\n"
        f"[green]1M+ papers indexed[/green]\n"
        f"[yellow]Powered by RAGGuard[/yellow]",
        border_style="cyan"
    ))

    # Connect to Qdrant
    try:
        from qdrant_client import QdrantClient
        from ragguard import QdrantSecureRetriever, load_policy
        from sentence_transformers import SentenceTransformer

        console.print(f"\n🔌 Connecting to Qdrant at {args.host}:{args.port}...")
        client = QdrantClient(host=args.host, port=args.port)

        # Check collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if args.collection not in collection_names:
            console.print(f"[red]❌ Collection '{args.collection}' not found[/red]")
            console.print(f"\nAvailable collections: {', '.join(collection_names)}")
            console.print(f"\nRun: python load_to_vectordb.py --db qdrant")
            return

        # Load policy
        console.print("📋 Loading access control policy...")
        policy = load_policy("policy.yaml")

        # Load embedding model
        console.print("📦 Loading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Create secure retriever
        retriever = QdrantSecureRetriever(
            client=client,
            collection=args.collection,
            policy=policy,
            embed_fn=model.encode
        )

        console.print("[green]✅ Ready![/green]")

        # Select user
        user = select_user()

        console.print(f"\n✅ Logged in as: [green]{user['name']}[/green]")
        if user.get("institution"):
            console.print(f"   Institution: {user['institution']}")
        console.print(f"   Roles: {', '.join(user['roles'])}")

        # Interactive query loop
        while True:
            console.print("\n" + "─" * 60)
            query = Prompt.ask("\n💬 Enter your query (or 'quit' to exit)")

            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n👋 Goodbye!")
                break

            if not query.strip():
                continue

            search_papers(retriever, query, user)

    except ImportError as e:
        console.print(f"[red]❌ Missing dependencies: {e}[/red]")
        console.print("\nInstall with: pip install -r requirements.txt")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    main()
