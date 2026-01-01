"""
Example: Health check endpoints for FastAPI applications.

This example shows how to integrate RAGGuard health checks with FastAPI
for Kubernetes deployment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from ragguard import Policy
from ragguard.retrievers import QdrantSecureRetriever
from ragguard.health import create_fastapi_health_endpoints

# Initialize FastAPI app
app = FastAPI(
    title="RAGGuard Search API",
    description="Secure vector search with access control",
    version="1.0.0"
)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
try:
    client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
except Exception:
    pass  # Collection already exists

# Load policy
policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {
            "name": "department-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }
    ],
    "default": "deny"
})

# Initialize secure retriever
retriever = QdrantSecureRetriever(
    client=client,
    collection="documents",
    policy=policy
)

# Register health check endpoints
create_fastapi_health_endpoints(app, retriever)


# Request/Response models
class SearchRequest(BaseModel):
    query: List[float]
    user: Dict[str, Any]
    limit: int = 10


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform secure vector search."""
    try:
        results = retriever.search(
            request.query,
            request.user,
            limit=request.limit
        )

        return SearchResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """API information."""
    return {
        "name": "RAGGuard Search API",
        "version": "1.0.0",
        "health_endpoints": {
            "liveness": "/health",
            "readiness": "/ready",
            "startup": "/startup"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI app with health endpoints:")
    print("  GET /health  - Liveness probe (is service alive?)")
    print("  GET /ready   - Readiness probe (can service accept traffic?)")
    print("  GET /startup - Startup probe (has service finished initializing?)")
    print("  GET /docs    - OpenAPI documentation")
    print()
    print("Kubernetes deployment example:")
    print("""
apiVersion: v1
kind: Pod
metadata:
  name: ragguard-app
spec:
  containers:
  - name: app
    image: my-ragguard-app:latest
    ports:
    - containerPort: 8000
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 30
      timeoutSeconds: 5
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    startupProbe:
      httpGet:
        path: /startup
        port: 8000
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 5
      failureThreshold: 30  # 30 * 5s = 150s max startup time
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000)
