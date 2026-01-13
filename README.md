# RAGGuard

**The security layer your RAG application is missing.**

[![PyPI](https://img.shields.io/badge/pypi-v0.3.1-blue)](https://pypi.org/project/ragguard/)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/ragguard/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Security](https://img.shields.io/badge/security-19%2F19%20attacks%20blocked-brightgreen.svg)]()

```
┌─────────────────────────────────────────────────────────────────┐
│                    BRING YOUR OWN PERMISSIONS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INLINE POLICIES    CUSTOM FILTERS    ACL DOCUMENTS    ENTERPRISE AUTH   │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│   │ rules:      │   │ class My    │   │ {"acl": {   │   │   OPA       │   │
│   │  - allow:   │   │   Filter:   │   │  "users":   │   │   Cerbos    │   │
│   │      dept   │   │   def build │   │  ["alice"]} │   │   OpenFGA   │   │
│   │             │   │   ...       │   │ }}          │   │   Permit.io │   │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│      Code/YAML        Full Control     Explicit Lists   Policy Engines   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

> **The Problem**: Your RAG system retrieves documents, then filters by permissions. But by then, unauthorized data has already been exposed to the retrieval layer. That's a data leak.
>
> > **The Solution**: RAGGuard filters **during** vector search, not after. Zero unauthorized exposure.
> >
> > **Works with any authorization system** - use your existing permissions infrastructure (OPA, Cerbos, OpenFGA, custom RBAC, ACLs) or define policies inline. RAGGuard translates your authorization decisions into vector database filters.
> >
> > ```
> > ┌────────────────────────────────────┬────────────────────────────────────┐
> > │         WITHOUT RAGGUARD           │          WITH RAGGUARD             │
> > ├────────────────────────────────────┼────────────────────────────────────┤
> > │                                    │                                    │
> > │   Vector Search                    │   Vector Search                    │
> > │   Returns 10 docs ─────────────┐   │   + Permission Filter              │
> > │   (includes unauthorized)      │   │   Returns 10 docs                  │
> > │                │               │   │   (all authorized)                 │
> > │                ▼               │   │           │                        │
> > │   Filter in Python             │   │           │                        │
> > │   Remove 7 docs                │   │           │                        │
> > │                │               │   │           ▼                        │
> > │                ▼               │   │                                    │
> > │   Return 3 docs                │   │   Return 10 docs                   │
> > │   ❌ Data leaked               │   │   ✅ Zero exposure                 │
> > │   ❌ Wrong count               │   │   ✅ Correct count                 │
> > │                                    │                                    │
> > └────────────────────────────────────┴────────────────────────────────────┘
> > ```
> >
> > ## Quick Start
> >
> > ```bash
> > pip install ragguard[chromadb]
> > ```
> >
> > ```python
> > import chromadb
> > from ragguard import ChromaDBSecureRetriever, Policy
> >
> > # Your existing ChromaDB setup
> > client = chromadb.Client()
> > collection = client.get_or_create_collection("docs")
> >
> > # Add documents with permission metadata
> > collection.add(
> >     documents=["Q4 financials show 20% growth", "Public product roadmap"],
> >     metadatas=[
> >         {"department": "finance", "classification": "confidential"},
> >         {"department": "product", "classification": "public"}
> >     ],
> >     ids=["doc1", "doc2"]
> > )
> >
> > # Define who can access what
> > policy = Policy(
> >     rules=[
> >         {"allow": {"department": "finance"}, "when": {"user.role": "finance_team"}},
> >         {"allow": {"classification": "public"}}  # Everyone can see public docs
> >     ]
> > )
> >
> > # Create secure retriever
> > retriever = ChromaDBSecureRetriever(
> >     collection=collection,
> >     policy=policy
> > )
> >
> > # Query with user context - only returns authorized documents
> > results = retriever.retrieve(
> >     query="quarterly results",
> >     user_context={"role": "finance_team"}
> > )
> > ```
> >
> > ## Supported Vector Databases
> >
> > | Database | Status | Installation |
> > |----------|--------|--------------|
> > | ChromaDB | ✅ | `pip install ragguard[chromadb]` |
> > | Qdrant | ✅ | `pip install ragguard[qdrant]` |
> > | Pinecone | ✅ | `pip install ragguard[pinecone]` |
> > | pgvector | ✅ | `pip install ragguard[pgvector]` |
> > | Weaviate | ✅ | `pip install ragguard[weaviate]` |
> > | Milvus | ✅ | `pip install ragguard[milvus]` |
> > | FAISS | ✅ | `pip install ragguard[faiss]` |
> > | OpenSearch | ✅ | `pip install ragguard[opensearch]` |
> > | Elasticsearch | ✅ | `pip install ragguard[elasticsearch]` |
> > | Redis | ✅ | `pip install ragguard[redis]` |
> > | LanceDB | ✅ | `pip install ragguard[lancedb]` |
> > | Vespa | ✅ | `pip install ragguard[vespa]` |
> > | Marqo | ✅ | `pip install ragguard[marqo]` |
> > | Typesense | ✅ | `pip install ragguard[typesense]` |
> >
> > ## Framework Integrations
> >
> > ```python
> > # LangChain
> > from ragguard.integrations import LangChainSecureRetriever
> >
> > retriever = LangChainSecureRetriever(
> >     vectorstore=your_vectorstore,
> >     policy=policy
> > )
> > chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
> >
> > # LlamaIndex
> > from ragguard.integrations import LlamaIndexSecureRetriever
> >
> > retriever = LlamaIndexSecureRetriever(
> >     index=your_index,
> >     policy=policy
> > )
> > ```
> >
> > ## Enterprise Authorization
> >
> > Connect to your existing policy engine:
> >
> > ```python
> > from ragguard.auth import OPAProvider, CerbosProvider, OpenFGAProvider
> >
> > # Open Policy Agent
> > policy = OPAProvider(url="http://localhost:8181/v1/data/ragguard/allow")
> >
> > # Cerbos
> > policy = CerbosProvider(url="http://localhost:3592")
> >
> > # OpenFGA
> > policy = OpenFGAProvider(
> >     api_url="http://localhost:8080",
> >     store_id="your-store-id"
> > )
> > ```
> >
> > ## Why RAGGuard?
> >
> > ### Security
> > - **Zero data exposure** - Unauthorized documents never leave the database
> > - - **Blocks 19/19 known RAG attack patterns** including prompt injection, context manipulation, and privilege escalation
> >   - - **Audit logging** - Full trail of who accessed what
> >    
> >     - ### Compliance
> >     - - **HIPAA** - Patient data isolation
> >       - - **SOC2** - Access control audit trails
> >         - - **GDPR** - Data minimization by design
> >           - - **Multi-tenant isolation** - Customer data never mixes
> >            
> >             - ### Performance
> >             - - **No post-retrieval filtering** - Get exactly k authorized results
> >               - - **Native database filters** - Leverages built-in database optimization
> >                 - - **Minimal latency overhead** - <5ms for permission translation
> >                  
> >                   - ## Documentation
> >                  
> >                   - - [Full Documentation](https://ragguard.dev/docs)
> >                     - - [Examples](./examples)
> >                       - - [Security Model](./docs/security.md)
> >                         - - [API Reference](./docs/api.md)
> >                          
> >                           - ## Contributing
> >                          
> >                           - We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
> >                          
> >                           - ## License
> >
> > Apache 2.0 - See [LICENSE](./LICENSE) for details.
