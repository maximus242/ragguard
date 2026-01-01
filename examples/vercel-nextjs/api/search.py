"""
RAGGuard search endpoint for Vercel serverless functions.

Optimized for serverless deployment with connection reuse and fast cold starts.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
from typing import Optional

# Lazy imports to speed up cold starts
_imports_loaded = False


def ensure_imports():
    """Lazy load heavy imports on first request."""
    global _imports_loaded, QdrantSecureRetriever, load_policy, RetryConfig, QdrantClient

    if not _imports_loaded:
        from ragguard import QdrantSecureRetriever, load_policy, RetryConfig
        from qdrant_client import QdrantClient
        _imports_loaded = True


# Singleton pattern for connection reuse across invocations
_retriever: Optional[any] = None


def get_retriever():
    """
    Get or create retriever instance.

    This singleton pattern ensures connections are reused across
    serverless invocations, reducing latency and connection overhead.
    """
    global _retriever

    if _retriever is None:
        ensure_imports()

        # Initialize Qdrant client
        qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = os.environ.get('QDRANT_API_KEY')
        collection = os.environ.get('QDRANT_COLLECTION', 'documents')

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=5  # Lower timeout for serverless
        )

        # Load policy (could be from env, file, or remote URL)
        policy_source = os.environ.get('POLICY_YAML')
        if policy_source:
            # Policy from environment variable
            from ragguard.policy import Policy
            policy = Policy.from_yaml(policy_source)
        else:
            # Default policy (for demo - replace with your policy)
            policy = load_policy("policy.yaml")

        # Configure retry for serverless (shorter timeouts)
        retry_config = RetryConfig(
            max_retries=2,  # Lower retries for faster failure
            initial_delay=0.05,
            max_delay=2.0,  # Cap at 2s (Vercel timeout is 10s on hobby)
            exponential_base=2,
            jitter=True
        )

        _retriever = QdrantSecureRetriever(
            client=client,
            collection=collection,
            policy=policy,
            retry_config=retry_config,
            enable_filter_cache=True  # Cache filters for better performance
        )

    return _retriever


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle search requests."""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return

            body = self.rfile.read(content_length)
            data = json.loads(body)

            # Extract parameters
            query = data.get('query')
            user = data.get('user', {})
            limit = data.get('limit', 10)

            # Validate input
            if not query:
                self._send_error(400, "Missing required parameter: query")
                return

            if not isinstance(user, dict):
                self._send_error(400, "user must be an object")
                return

            if not isinstance(limit, int) or limit < 1 or limit > 100:
                self._send_error(400, "limit must be between 1 and 100")
                return

            # Execute permission-aware search
            retriever = get_retriever()
            results = retriever.search(
                query=query,
                user=user,
                limit=limit
            )

            # Convert results to JSON-serializable format
            results_json = []
            for r in results:
                result_dict = {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload if hasattr(r, 'payload') else {}
                }
                results_json.append(result_dict)

            # Send response
            self._send_json(200, {
                "success": True,
                "results": results_json,
                "count": len(results_json),
                "query": query,
                "user_id": user.get('id', 'unknown')
            })

        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            # Log error (appears in Vercel logs)
            print(f"ERROR: {str(e)}")
            self._send_error(500, f"Search failed: {str(e)}")

    def do_GET(self):
        """Health check endpoint."""
        self._send_json(200, {
            "status": "healthy",
            "service": "ragguard-search",
            "version": "0.3.0"
        })

    def _send_json(self, status_code: int, data: dict):
        """Send JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_json(status_code, {
            "success": False,
            "error": message
        })
