"""
Example: Health check endpoints for Flask applications.

This example shows how to integrate RAGGuard health checks with Flask
for Kubernetes deployment.
"""

from flask import Flask, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from ragguard import Policy
from ragguard.retrievers import QdrantSecureRetriever
from ragguard.health import create_flask_health_endpoints

# Initialize Flask app
app = Flask(__name__)

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
create_flask_health_endpoints(app, retriever)


@app.route("/search", methods=["POST"])
def search():
    """Example search endpoint."""
    from flask import request

    data = request.json
    query_vector = data.get("query")
    user_context = data.get("user")
    limit = data.get("limit", 10)

    results = retriever.search(query_vector, user_context, limit=limit)

    return jsonify({
        "results": results,
        "count": len(results)
    })


if __name__ == "__main__":
    print("Starting Flask app with health endpoints:")
    print("  GET /health  - Liveness probe (is service alive?)")
    print("  GET /ready   - Readiness probe (can service accept traffic?)")
    print("  GET /startup - Startup probe (has service finished initializing?)")
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
    - containerPort: 5000
    livenessProbe:
      httpGet:
        path: /health
        port: 5000
      initialDelaySeconds: 10
      periodSeconds: 30
      timeoutSeconds: 5
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /ready
        port: 5000
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    startupProbe:
      httpGet:
        path: /startup
        port: 5000
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 5
      failureThreshold: 30  # 30 * 5s = 150s max startup time
    """)

    app.run(host="0.0.0.0", port=5000)
