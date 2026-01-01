# Installation

## Requirements

- Python 3.9 or higher
- pip or poetry

## Basic Installation

```bash
pip install ragguard
```

## Installation with Backends

RAGGuard supports multiple vector and graph databases. Install with the backends you need:

### Vector Databases

```bash
# Qdrant
pip install ragguard[qdrant]

# ChromaDB
pip install ragguard[chromadb]

# PostgreSQL with pgvector
pip install ragguard[pgvector]

# Pinecone
pip install ragguard[pinecone]

# Weaviate
pip install ragguard[weaviate]

# Milvus
pip install ragguard[milvus]

# Elasticsearch / OpenSearch
pip install ragguard[elasticsearch]

# Azure Cognitive Search
pip install ragguard[azure]

# FAISS (local)
pip install ragguard[faiss]
```

### Graph Databases

```bash
# Neo4j
pip install ragguard[neo4j]

# Amazon Neptune (Gremlin)
pip install ragguard[neptune]

# TigerGraph
pip install ragguard[tigergraph]

# ArangoDB
pip install ragguard[arangodb]
```

### Multiple Backends

```bash
# Multiple specific backends
pip install ragguard[qdrant,chromadb,neo4j]

# All backends
pip install ragguard[all]
```

## Framework Integrations

```bash
# LangChain
pip install ragguard[langchain]

# LlamaIndex
pip install ragguard[llamaindex]

# CrewAI
pip install ragguard[crewai]

# AutoGen
pip install ragguard[autogen]

# DSPy
pip install ragguard[dspy]
```

## Development Installation

```bash
# Clone the repository
git clone https://github.com/maximus242/ragguard.git
cd ragguard

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Verifying Installation

```python
import ragguard
print(ragguard.__version__)

# Quick test
from ragguard import Policy
policy = Policy.from_dict({
    "version": "1",
    "rules": [{"name": "test", "allow": {"everyone": True}}],
    "default": "deny"
})
print("RAGGuard installed successfully!")
```

