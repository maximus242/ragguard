# FAQ & Known Limitations

## Frequently Asked Questions

### How does RAGGuard compare to post-filtering?

RAGGuard's primary advantage is **security** - it never exposes unauthorized documents, even temporarily. Post-filtering retrieves unauthorized docs then filters them out, which is a data leak.

For performance: At production scale (1M+ docs), RAGGuard is 2-3x faster. At small scale (< 10K docs), post-filtering may be faster, but you trade speed for security.

### Does this work with OpenAI embeddings?

Yes! RAGGuard works with any embeddings (OpenAI, Cohere, Sentence Transformers, custom models). Just pass your vectors to `search()`.

### Can I use this in production?

Yes. RAGGuard is production-ready with a comprehensive test suite including security, concurrency, and stability testing.

### What about async support?

Yes! RAGGuard provides async retrievers for all major backends.

```python
from ragguard.retrievers_async import AsyncQdrantSecureRetriever

async def search_docs(query, user):
    results = await retriever.search(query, user, limit=10)
    return results
```

### Does RAGGuard support multi-tenancy?

Yes! See the [use cases](use-cases.md#multi-tenant-saas) for examples.

### How do I debug policy issues?

Enable audit logging and use `PolicyEngine.evaluate()` to test specific user/document combinations.

## Known Limitations

### Policy Language

| Feature | Status | Workaround |
|---------|--------|------------|
| Regex matching | Planned v0.4.0 | Use exact matches or `in` operator |
| Date/time operations | Planned v0.4.0 | Store as Unix timestamps |
| Functions | Planned v0.4.0 | Pre-compute and store as fields |

### FAISS Limitations

FAISS doesn't support metadata filtering natively, so RAGGuard uses **post-filtering**:

- Slightly slower than native filtering backends
- May return fewer results than `limit` for restrictive policies
- Requires tuning `over_fetch_factor` parameter

**Recommendation:** For production, use Qdrant, pgvector, Weaviate, Pinecone, or ChromaDB.

### What We're NOT Building

To keep RAGGuard focused, we're **intentionally not supporting**:

- **General-purpose policy engine** - Use OPA/Cedar for that
- **API-level authorization** - Use Auth0/Authz for that
- **User authentication** - Use your existing auth system
- **Policy management UI** - Use your existing admin tools

**RAGGuard does one thing well:** Document-level permissions for vector search.

## Operator Support

| Operator | PolicyEngine | Native DB Filtering |
|----------|-------------|-------------------|
| `==`, `!=` | ✅ | ✅ |
| `in`, `not in` | ✅ | ✅ |
| `exists`, `not exists` | ✅ | ✅ |
| `>`, `<`, `>=`, `<=` | ✅ | ✅ |
| `OR`, `AND` | ✅ | ✅ |

All operators are fully supported with native database filtering.
