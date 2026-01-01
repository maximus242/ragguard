# Supported Backends

RAGGuard works with all major vector and graph databases (14 backends).

## Vector Databases

| Database | Install | Best For | Filtering |
|----------|---------|----------|-----------|
| **ChromaDB** | `pip install ragguard[chromadb]` | Local development | Native |
| **FAISS** | `pip install ragguard[faiss]` | Research/prototyping | Post-filter* |
| **Pinecone** | `pip install ragguard[pinecone]` | Managed cloud | Native |
| **Qdrant** | `pip install ragguard[qdrant]` | Production deployments | Native |
| **Weaviate** | `pip install ragguard[weaviate]` | GraphQL integration | Native |
| **pgvector** | `pip install ragguard[pgvector]` | PostgreSQL users | Native |
| **Milvus/Zilliz** | `pip install ragguard[milvus]` | Large-scale deployments | Native |
| **Elasticsearch** | `pip install ragguard[elasticsearch]` | Existing ES infrastructure | Native |
| **OpenSearch** | `pip install ragguard[opensearch]` | AWS-native deployments | Native |
| **Azure AI Search** | `pip install ragguard[azure]` | Azure cloud | Native |

## Graph Databases

| Database | Install | Best For | Filtering |
|----------|---------|----------|-----------|
| **Neo4j** | `pip install ragguard[neo4j]` | Knowledge graphs | Cypher |
| **Amazon Neptune** | `pip install ragguard[neptune]` | AWS graph workloads | Gremlin |
| **TigerGraph** | `pip install ragguard[tigergraph]` | Enterprise analytics | GSQL |
| **ArangoDB** | `pip install ragguard[arangodb]` | Multi-model databases | AQL |

> **\*FAISS Note:** FAISS doesn't support native metadata filtering. RAGGuard uses post-filtering for FAISS, providing security but not the performance benefits.

## Usage Examples

### ChromaDB

```python
import chromadb
from ragguard import ChromaDBSecureRetriever, load_policy

client = chromadb.Client()
collection = client.get_or_create_collection("my_docs")

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=load_policy("policy.yaml"),
    embed_fn=embeddings.embed_query  # Optional
)

results = retriever.search(
    query="company policies",
    user={"department": "engineering"},
    limit=10
)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy

client = QdrantClient("localhost", port=6333)

retriever = QdrantSecureRetriever(
    client=client,
    collection="documents",
    policy=load_policy("policy.yaml")
)

results = retriever.search(
    query=query_vector,
    user={"department": "engineering"},
    limit=10
)
```

### Pinecone

```python
from pinecone import Pinecone
from ragguard import PineconeSecureRetriever, load_policy

pc = Pinecone(api_key="your-api-key")
index = pc.Index("your-index-name")

retriever = PineconeSecureRetriever(
    index=index,
    policy=load_policy("policy.yaml"),
    namespace="prod"  # Optional
)

results = retriever.search(
    query="customer support tickets",
    user={"id": "alice", "roles": ["support"]},
    limit=10
)
```

### pgvector

```python
import psycopg2
from ragguard import PgvectorSecureRetriever, load_policy

conn = psycopg2.connect("postgresql://localhost/mydb")

retriever = PgvectorSecureRetriever(
    connection=conn,
    table="documents",
    policy=load_policy("policy.yaml")
)

results = retriever.search(
    query=query_vector,
    user={"department": "finance"},
    limit=10
)
```

### Weaviate

```python
import weaviate
from ragguard import WeaviateSecureRetriever, load_policy

client = weaviate.Client("http://localhost:8080")

retriever = WeaviateSecureRetriever(
    client=client,
    collection="Documents",
    policy=load_policy("policy.yaml")
)

results = retriever.search(
    query="product documentation",
    user={"department": "product"},
    limit=10
)
```

### FAISS

```python
import faiss
from ragguard import FAISSSecureRetriever, load_policy

# Create FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)
index.add(vectors)  # Your embeddings

# Metadata stored separately
metadata = [
    {"id": 0, "department": "eng", "confidential": False},
    {"id": 1, "department": "finance", "confidential": True},
]

retriever = FAISSSecureRetriever(
    index=index,
    metadata=metadata,
    policy=load_policy("policy.yaml"),
    over_fetch_factor=3  # Fetch 3x for filtering
)
```

> **FAISS Limitations:** FAISS uses post-filtering. Results may be less than `limit` for restrictive policies. For production at scale, prefer backends with native filtering.
