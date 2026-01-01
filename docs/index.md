# RAGGuard Documentation

**The security layer your RAG application is missing.**

RAGGuard filters documents **during** vector search, not after. Zero unauthorized exposure.

## Quick Start

```python
pip install ragguard[chromadb]
```

```python
from ragguard import ChromaDBSecureRetriever, Policy

# Define access control policy
policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {"name": "dept-access", "allow": {"conditions": ["user.department == document.department"]}},
        {"name": "public", "match": {"visibility": "public"}, "allow": {"everyone": True}}
    ],
    "default": "deny"
})

# Create secure retriever
retriever = ChromaDBSecureRetriever(
    client=chromadb.Client(),
    collection="documents",
    policy=policy
)

# Search with user context - only authorized documents returned
results = retriever.search(
    query="quarterly report",
    user={"id": "alice", "department": "engineering"},
    limit=10
)
```

## Documentation Sections

### Getting Started
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Core Concepts](getting-started/concepts.md)

### Policies
- [Policy Syntax](policies/syntax.md)
- [Conditions](policies/conditions.md)
- [Examples](policies/examples.md)

### Vector Databases
- [Qdrant](backends/qdrant.md)
- [ChromaDB](backends/chromadb.md)
- [pgvector](backends/pgvector.md)
- [Pinecone](backends/pinecone.md)
- [Weaviate](backends/weaviate.md)
- [Milvus](backends/milvus.md)
- [Elasticsearch](backends/elasticsearch.md)
- [FAISS](backends/faiss.md)

### Graph Databases
- [Neo4j](backends/neo4j.md)
- [Amazon Neptune](backends/neptune.md)
- [TigerGraph](backends/tigergraph.md)
- [ArangoDB](backends/arangodb.md)

### Framework Integrations
- [LangChain](integrations/langchain.md)
- [LangGraph](integrations/langgraph.md)
- [LlamaIndex](integrations/llamaindex.md)
- [CrewAI](integrations/crewai.md)
- [AutoGen](integrations/autogen.md)
- [DSPy](integrations/dspy.md)
- [MCP](integrations/mcp.md)
- [A2A Protocol](integrations/a2a.md)
- [OpenAI Assistants](integrations/openai-assistants.md)

### Advanced Features
- [Caching](advanced/caching.md)
- [Circuit Breaker](advanced/circuit-breaker.md)
- [Health Checks](advanced/health-checks.md)
- [Audit Logging](advanced/audit-logging.md)
- [Async Support](advanced/async.md)

### API Reference
- [Policy Engine](api/policy-engine.md)
- [Retrievers](api/retrievers.md)
- [Filter Builders](api/filter-builders.md)

## Why RAGGuard?

### The Problem

Traditional RAG systems retrieve documents first, then filter by permissions:

```
User Query → Vector Search → Filter Results → Return
                    ↓
         Unauthorized data exposed to retrieval layer
```

### The Solution

RAGGuard filters **during** the vector search:

```
User Query → Policy Filter + Vector Search → Return
                    ↓
         Only authorized documents ever retrieved
```

## Key Features

| Feature | Description |
|---------|-------------|
| **14 Vector DBs** | Qdrant, ChromaDB, pgvector, Pinecone, Weaviate, Milvus, Elasticsearch, OpenSearch, Azure Search, FAISS, Neo4j, Neptune, TigerGraph, ArangoDB |
| **10 Integrations** | LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, DSPy, MCP, A2A, OpenAI Assistants, AWS Bedrock |
| **Production Ready** | Comprehensive test suite, circuit breaker, caching, health checks |

## Installation

```bash
# Core package
pip install ragguard

# With specific backends
pip install ragguard[qdrant]
pip install ragguard[chromadb]
pip install ragguard[pgvector]

# With all backends
pip install ragguard[all]
```

## Support

- GitHub Issues: [github.com/maximus242/ragguard/issues](https://github.com/maximus242/ragguard/issues)
- Documentation: This site
