# RAGGuard

**The security layer your RAG application is missing.**

[![PyPI version](https://img.shields.io/pypi/v/ragguard.svg)](https://pypi.org/project/ragguard/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/ragguard.svg)](https://pypi.org/project/ragguard/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Security](https://img.shields.io/badge/security-19%2F19%20attacks%20blocked-brightgreen.svg)]()

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BRING YOUR OWN PERMISSIONS                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INLINE POLICIES     CUSTOM FILTERS     ACL DOCUMENTS     ENTERPRISE AUTH   │
│   ┌─────────────┐     ┌─────────────┐    ┌────────────┐    ┌─────────────┐   │
│   │ rules:      │     │ class My    │    │ {"acl": {  │    │    OPA      │   │
│   │  - allow:   │     │   Filter:   │    │   "users": │    │   Cerbos    │   │
│   │     dept    │     │   def build │    │   ["alice"]│    │   OpenFGA   │   │
│   │             │     │     ...     │    │  }}        │    │   Permit.io │   │
│   └─────────────┘     └─────────────┘    └────────────┘    └─────────────┘   │
│     Code/YAML           Full Control      Explicit Lists    Policy Engines   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **The Problem**: Your RAG system retrieves documents, then filters by permissions. But by then, unauthorized data has already been exposed to the retrieval layer. That's a data leak.
>
> **The Solution**: RAGGuard filters **during** vector search, not after. Zero unauthorized exposure.

**Works with any authorization system** - use your existing permissions infrastructure (OPA, Cerbos, OpenFGA, custom RBAC, ACLs) or define policies inline. RAGGuard translates your authorization decisions into vector database filters.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   WITHOUT RAGGUARD                      WITH RAGGUARD                       │
├─────────────────────────────────────────────────────────────────────────────┤
│   Vector Search                         Vector Search                       │
│   Returns 10 docs ──────────┐           + Permission Filter                 │
│   (includes unauthorized)   │           Returns 10 docs                     │
│             │               │           (all authorized)                    │
│             ▼               │                  │                            │
│   Filter in Python          │                  │                            │
│   Remove 7 docs             │                  │                            │
│             │               │                  │                            │
│             ▼               │                  ▼                            │
│   Return 3 docs             │           Return 10 docs                      │
│   ❌ Data leaked            │           ✅ Zero exposure                    │
│   ❌ Wrong count            │           ✅ Correct count                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install ragguard[chromadb]
```

```python
import chromadb
from ragguard import ChromaDBSecureRetriever, Policy

# 1. Your existing ChromaDB setup
client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(
    ids=["1", "2", "3"],
    documents=["Finance Report", "Engineering Doc", "Public Blog"],
    metadatas=[
        {"department": "finance", "confidential": True},
        {"department": "engineering", "confidential": False},
        {"department": "public", "confidential": False}
    ]
)

# 2. Define access policy
policy = Policy.from_dict({
    "version": "1",
    "rules": [
        {"name": "same-dept", "allow": {"conditions": ["user.department == document.department"]}},
        {"name": "public", "match": {"confidential": False}, "allow": {"everyone": True}}
    ],
    "default": "deny"
})

# 3. Search with automatic permission filtering
retriever = ChromaDBSecureRetriever(collection=collection, policy=policy)

results = retriever.search(
    query="quarterly report",
    user={"id": "alice", "department": "finance"},
    limit=10
)
# Alice sees finance docs + public docs only
```

**That's it.** Documents are filtered at the database level. No post-filtering. No data leaks.

## Bring Your Own Authorization

RAGGuard doesn't force you into a specific permissions model. Use what you already have:

### Option 1: Inline Policies (shown above)
Define policies directly in code or YAML - great for getting started or simple use cases.

### Option 2: Custom Filter Builders
Plug in any authorization logic with full control:

```python
from ragguard.filters import CustomFilterBuilder

class MyAuthFilter(CustomFilterBuilder):
    def build_filter(self, policy, user, backend):
        # Query your auth system, check ACLs, call APIs - whatever you need
        allowed_docs = my_auth_service.get_accessible_docs(user["id"])
        return {"doc_id": {"$in": allowed_docs}}

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    custom_filter_builder=MyAuthFilter()
)
```

### Option 3: ACL-Based Documents
For documents with explicit access control lists:

```python
from ragguard.filters import ACLFilterBuilder

# Documents have: {"acl": {"users": ["alice"], "groups": ["eng"], "public": false}}
retriever = QdrantSecureRetriever(
    collection=collection,
    policy=policy,
    custom_filter_builder=ACLFilterBuilder(
        get_user_groups=lambda user: fetch_groups_from_ldap(user["id"])
    )
)
```

### Option 4: Enterprise Authorization Systems
Connect to dedicated authorization services (available in `ragguard-enterprise`):

| System | Description |
|--------|-------------|
| **OPA** | Open Policy Agent - policy as code |
| **Cerbos** | Access control for cloud-native apps |
| **OpenFGA** | Google Zanzibar-inspired fine-grained auth |
| **Permit.io** | Permissions as a service |
| **Auth0/Okta** | Identity provider integration |

## Supported Backends

| Vector DBs | Graph DBs |
|------------|-----------|
| Qdrant, ChromaDB, Pinecone, pgvector, Weaviate, Milvus, FAISS, Elasticsearch, OpenSearch, Azure AI Search | Neo4j, Neptune, TigerGraph, ArangoDB |

## Integrations

LangChain • LlamaIndex • LangGraph • CrewAI • DSPy • AWS Bedrock

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and basic setup |
| [Policy Format](docs/policy-format.md) | Policy syntax and operators |
| [Backends](docs/backends.md) | Database-specific examples |
| [Integrations](docs/integrations.md) | LangChain, LlamaIndex, etc. |
| [Production](docs/production.md) | Health checks, logging, async |
| [Kubernetes](docs/kubernetes.md) | K8s deployment guide |
| [Security](docs/security.md) | Security testing & guarantees |
| [Use Cases](docs/use-cases.md) | Multi-tenant, healthcare, etc. |
| [FAQ](docs/faq.md) | Common questions & limitations |

## Installation

```bash
# With a specific backend
pip install ragguard[qdrant]
pip install ragguard[chromadb]
pip install ragguard[pgvector]
pip install ragguard[pinecone]

# With framework integration
pip install ragguard[langchain]
pip install ragguard[llamaindex]

# Everything
pip install ragguard[all]
```

**Python Compatibility**: Fully tested on Python 3.9-3.13. Python 3.14 has limited support due to upstream dependencies (chromadb, langchain) not yet supporting Python 3.14.

## Why RAGGuard?

| Challenge | Without RAGGuard | With RAGGuard |
|-----------|------------------|---------------|
| Data leaks | Filter after retrieval = data exposed | Filter during search = zero exposure |
| Authorization | Rebuild permission logic for RAG | Plug in your existing auth system |
| Multi-database | Custom filter code per DB | One integration, 14 databases |
| Setup time | Days/weeks | 5 minutes |
| Security testing | DIY | Comprehensive test suite |

## License

Apache-2.0 - See [LICENSE](LICENSE) for details.

---

**Built for the RAG community** • [Examples](examples/) • [GitHub](https://github.com/maximus242/ragguard)
