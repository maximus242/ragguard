# Getting Started

## Quick Start (Production Setup)

### 1. Install

```bash
pip install ragguard[chromadb]  # or pinecone, faiss, qdrant, etc.
```

### 2. Define Policy

Create `policy.yaml`:

```yaml
version: "1"

rules:
  # Public documents - everyone can see
  - name: "public-docs"
    match:
      public: true
    allow:
      everyone: true

  # Department documents - same department only
  - name: "department-docs"
    allow:
      conditions:
        - user.department == document.department

  # Admin gets everything
  - name: "admin-access"
    allow:
      roles: ["admin"]

default: deny
```

### 3. Add 5 Lines of Code

```python
import chromadb
from ragguard import ChromaDBSecureRetriever, load_policy

# Your existing ChromaDB setup
client = chromadb.Client()
collection = client.get_or_create_collection("docs")

# Wrap with RAGGuard (3 lines)
policy = load_policy("policy.yaml")
retriever = ChromaDBSecureRetriever(collection=collection, policy=policy)

# Search with automatic permission filtering (1 line)
results = retriever.search(
    query="What is our Q3 revenue?",
    user={"id": "alice", "department": "finance", "roles": ["analyst"]},
    limit=10
)
# Returns only documents alice is authorized to see
```

**That's it!** Every search now enforces permissions automatically.

## Installation Options

```bash
# Minimal (no vector DB dependencies)
pip install ragguard

# Specific vector database
pip install ragguard[chromadb]
pip install ragguard[faiss]
pip install ragguard[pinecone]
pip install ragguard[qdrant]
pip install ragguard[weaviate]
pip install ragguard[pgvector]

# Framework integrations
pip install ragguard[langchain]
pip install ragguard[llamaindex]
pip install ragguard[langgraph]
pip install ragguard[crewai]

# Everything
pip install ragguard[all]

# Development
pip install ragguard[dev]
```

## Testing Your Setup

```python
from ragguard.policy import PolicyEngine

engine = PolicyEngine(policy)

# Test specific user/document combinations
alice = {"id": "alice", "department": "eng", "roles": ["engineer"]}
doc = {"department": "eng", "confidential": False}

can_access = engine.evaluate(alice, doc)
assert can_access == True  # Alice can access eng docs
```
