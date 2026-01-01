# Kubernetes Deployment

Deploy RAGGuard applications to production with Kubernetes.

## Quick Start

```bash
cd k8s/examples/

# Deploy with Qdrant backend
kubectl apply -f fastapi-qdrant.yaml

# Deploy with pgvector backend
kubectl apply -f fastapi-pgvector.yaml
```

## Production-Ready Features

- **Health Checks**: Liveness, readiness, and startup probes
- **Auto-Scaling**: HorizontalPodAutoscaler based on CPU/memory
- **High Availability**: Multiple replicas with Pod Disruption Budgets
- **Connection Pooling**: Optimized database connection management
- **Resource Management**: CPU/memory limits and requests
- **Security**: Non-root containers, secret management

## Health Check Endpoints

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health/liveness")
async def liveness():
    """Returns 200 if application is alive"""
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    """Returns 200 if ready to serve traffic"""
    health = retriever.health_check()
    if not health["healthy"]:
        raise HTTPException(status_code=503)
    return health
```

## Probe Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

## Example Deployment

The `k8s/examples/` directory includes:

- **app.py**: Production FastAPI application with RAGGuard
- **Dockerfile**: Multi-stage optimized container image
- **fastapi-qdrant.yaml**: Complete Qdrant deployment
- **fastapi-pgvector.yaml**: PostgreSQL/pgvector with connection pooling
- **README.md**: Comprehensive deployment guide
