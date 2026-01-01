# Changelog

All notable changes to RAGGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-01-01

### Added

- **Async graph database retrievers**
  - `AsyncNeptuneSecureRetriever` - Async Neptune/Gremlin support
  - `AsyncTigerGraphSecureRetriever` - Async TigerGraph/GSQL support
  - `AsyncArangoDBSecureRetriever` - Async ArangoDB/AQL support
  - `AsyncNeo4jSecureRetriever` - Async Neo4j/Cypher support
  - `AsyncGraphSecureRetrieverBase` - Base class for async graph retrievers

- **Additional async vector retrievers**
  - `AsyncMilvusSecureRetriever` - Async Milvus/Zilliz support
  - `AsyncElasticsearchSecureRetriever` - Async Elasticsearch support
  - `AsyncOpenSearchSecureRetriever` - Async OpenSearch support
  - `AsyncAzureSearchSecureRetriever` - Async Azure AI Search support

### Fixed

- Circuit breaker tests now use time mocking instead of `time.sleep()` for reliability
- Strengthened weak assertions in validation tests (explicit `None` checks)
- Fixed `datetime.utcnow()` deprecation warning in `TimeBasedResolver`
- Fixed `pytest.importorskip` deprecation warnings for chromadb imports
- Added consistent circuit breaker and validation parameters across all sync retrievers

### Changed

- Reduced test warnings from 629 to 0 by filtering expected third-party warnings
- Improved test reliability by removing time-dependent flaky tests

## [0.3.0] - 2025-12-29

### Added

- **SecureRetrieverConfig dataclass** for consolidating retriever constructor parameters
  - All 14 retrievers now accept an optional `config` parameter
  - Simplifies retriever instantiation with many settings
  - Config values override individual parameters when provided

- **Comprehensive permission system integrations**
  - Cerbos integration (`ragguard.iam.cerbos`)
  - OpenFGA integration (`ragguard.iam.openfga`)
  - AWS IAM integration (`ragguard.iam.aws_iam`)
  - Google Cloud IAM integration (`ragguard.iam.google`)
  - OPA (Open Policy Agent) integration (`ragguard.iam.opa`)

- **Graph database retrievers**
  - Neo4j retriever (`ragguard.retrievers.neo4j`)
  - Amazon Neptune retriever (`ragguard.retrievers.neptune`)
  - TigerGraph retriever (`ragguard.retrievers.tigergraph`)
  - ArangoDB retriever (`ragguard.retrievers.arangodb`)

- **Framework integrations**
  - LangChain retriever wrapper (`ragguard.integrations.langchain`)
  - LlamaIndex retriever wrapper (`ragguard.integrations.llamaindex`)
  - CrewAI integration (`ragguard.integrations.crewai`)
  - LangGraph integration (`ragguard.integrations.langgraph`)
  - MCP server integration (`ragguard.integrations.mcp`)

- **FilterResult class** with explicit semantics for permission outcomes
  - `FilterResult.allow_all()` - User has access to all documents
  - `FilterResult.deny_all()` - User has no access
  - `FilterResult.conditional()` - User has conditional access with filter

- **OR/AND expression support** in policy conditions (v0.3.0)
  - Complex boolean logic in access control rules
  - Expression depth and branch limits for DoS prevention

### Changed

- **BREAKING**: Converted getter methods to properties for cleaner API
  - `FilterResult.is_deny_all()` → `FilterResult.is_deny_all`
  - `FilterResult.is_allow_all()` → `FilterResult.is_allow_all`
  - `FilterResult.is_conditional()` → `FilterResult.is_conditional`
  - `AllowConditions.is_empty()` → `AllowConditions.is_empty`
  - `BaseSecureRetriever.get_cache_stats()` → `BaseSecureRetriever.cache_stats`
  - `BaseSecureRetriever.get_circuit_breaker_stats()` → `BaseSecureRetriever.circuit_breaker_stats`
  - Backwards-compatible method aliases are provided but deprecated

- Improved policy validation with semantic checks and warnings
- Enhanced error messages for policy compilation failures
- Consolidated test suite (removed 35 redundant test files)

### Fixed

- License classifier inconsistency (now correctly shows Apache-2.0)
- Removed legacy backup file (compiler.py.backup)

### Security

- Added PolicyLimits class with DoS prevention limits:
  - MAX_RULES = 100
  - MAX_CONDITIONS_PER_RULE = 100
  - MAX_LIST_SIZE = 1000
  - MAX_POLICY_SIZE_BYTES = 1MB
  - MAX_EXPRESSION_DEPTH = 10
  - MAX_EXPRESSION_BRANCHES = 50

## [0.2.0] - 2025-12-01

### Added

- Initial public release
- Core permission-aware retrieval functionality
- Support for Qdrant, ChromaDB, Pinecone, pgvector, Milvus, Weaviate, FAISS, Elasticsearch, OpenSearch, Azure AI Search
- YAML-based policy definition
- Filter caching with LRU eviction
- Circuit breaker pattern for resilience
- Audit logging
- Metrics collection (Prometheus, StatsD, CloudWatch)

