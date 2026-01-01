"""
Example FastAPI application with RAGGuard for Kubernetes deployment.

This application demonstrates how to integrate RAGGuard with FastAPI,
including health checks, connection pooling, and proper lifecycle management.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging

# RAGGuard imports
from ragguard import Policy
from ragguard.retrievers import QdrantSecureRetriever, PgvectorSecureRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAGGuard Secure API",
    description="Production-ready RAG application with access control",
    version="1.0.0"
)

# Global retriever instance (initialized on startup)
retriever = None


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    user_id: str
    department: Optional[str] = None
    roles: Optional[List[str]] = None
    limit: int = 10


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


# Startup event - initialize retriever with connection pooling
@app.on_event("startup")
async def startup_event():
    """Initialize RAGGuard retriever on application startup."""
    global retriever

    logger.info("Initializing RAGGuard retriever...")

    # Load policy
    policy_path = os.getenv("RAGGUARD_POLICY_PATH", "/etc/ragguard/policy.yaml")
    policy = Policy.from_file(policy_path)

    # Initialize retriever based on backend type
    backend = os.getenv("BACKEND_TYPE", "qdrant")

    if backend == "qdrant":
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")

        client = QdrantClient(url=qdrant_url)

        retriever = QdrantSecureRetriever(
            client=client,
            collection_name=collection_name,
            policy=policy,
            enable_filter_cache=os.getenv("RAGGUARD_ENABLE_CACHE", "true").lower() == "true",
            filter_cache_size=int(os.getenv("RAGGUARD_CACHE_SIZE", "2000")),
            enable_retry=os.getenv("RAGGUARD_ENABLE_RETRY", "true").lower() == "true",
            enable_validation=os.getenv("RAGGUARD_ENABLE_VALIDATION", "true").lower() == "true"
        )

    elif backend == "pgvector":
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable required for pgvector backend")

        # Create engine with connection pooling
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=int(os.getenv("PGVECTOR_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("PGVECTOR_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("PGVECTOR_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("PGVECTOR_POOL_RECYCLE", "3600")),
            pool_pre_ping=True  # Verify connections before using
        )

        retriever = PgvectorSecureRetriever(
            connection=engine,
            collection="documents",
            policy=policy,
            enable_filter_cache=os.getenv("RAGGUARD_ENABLE_CACHE", "true").lower() == "true",
            filter_cache_size=int(os.getenv("RAGGUARD_CACHE_SIZE", "2000")),
            enable_retry=os.getenv("RAGGUARD_ENABLE_RETRY", "true").lower() == "true",
            enable_validation=os.getenv("RAGGUARD_ENABLE_VALIDATION", "true").lower() == "true"
        )

    else:
        raise ValueError(f"Unsupported backend type: {backend}")

    logger.info(f"RAGGuard retriever initialized with {backend} backend")


# Shutdown event - cleanup resources
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    global retriever

    if retriever:
        logger.info("Closing retriever connections...")
        # Use context manager cleanup if available
        if hasattr(retriever, '__exit__'):
            retriever.__exit__(None, None, None)
        logger.info("Retriever connections closed")


# Health check endpoints for Kubernetes probes
@app.get("/health/liveness")
async def liveness():
    """
    Liveness probe - checks if the application is running.

    Returns 200 if the application is alive, 503 otherwise.
    This is used by Kubernetes to restart the container if it's unhealthy.
    """
    return {"status": "alive"}


@app.get("/health/readiness")
async def readiness():
    """
    Readiness probe - checks if the application can serve traffic.

    Returns 200 if ready to serve requests, 503 otherwise.
    This is used by Kubernetes to route traffic to the container.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # Check backend health
    try:
        health_status = retriever.health_check()
        if not health_status.get("healthy", False):
            raise HTTPException(
                status_code=503,
                detail=f"Backend unhealthy: {health_status.get('error', 'Unknown error')}"
            )

        return {
            "status": "ready",
            "backend": health_status.get("backend"),
            "details": health_status.get("details", {})
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/health/startup")
async def startup():
    """
    Startup probe - checks if the application has finished starting up.

    Returns 200 when initialization is complete, 503 otherwise.
    This gives slow-starting containers more time before liveness checks begin.
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Application still starting up")

    return {"status": "started"}


# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search documents with permission-based access control.

    Args:
        request: Search request with query and user context

    Returns:
        Filtered search results based on user permissions
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    # Build user context
    user = {
        "id": request.user_id,
        "department": request.department,
        "roles": request.roles or []
    }

    try:
        # Execute search with permission filtering
        results = retriever.search(
            query=request.query,
            user=user,
            limit=request.limit
        )

        return SearchResponse(
            results=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "RAGGuard Secure API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "search": "/search",
            "health": {
                "liveness": "/health/liveness",
                "readiness": "/health/readiness",
                "startup": "/health/startup"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
