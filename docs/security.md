# Security & Testing

RAGGuard has been extensively tested for production use.

## Security Testing

**19 attack scenarios blocked** (100% pass rate):

- SQL/NoSQL injection
- Type confusion attacks
- Permission escalation attempts
- Cache poisoning
- Null/None bypasses

**Zero unauthorized documents** in all test scenarios.

**Adversarial input handling:**
- 1MB+ query strings
- Unicode/emoji attacks
- Null bytes and control characters
- Circular references

## Scale & Reliability Testing

- **100 concurrent users**: Zero permission leaks, 116 queries/sec
- **1000-query stability**: Zero errors, 99.5% cache hit rate
- **Multi-tenant isolation**: Perfect tenant boundaries verified
- **Policy updates under load**: Hot updates without downtime

## Performance

- **5.72µs overhead** (p50) - negligible impact
- **99.9% cache hit rate** in production scenarios
- **No memory leaks** over extended runs

## Running the Test Suite

```bash
# Full test suite
pip install -e ".[dev]"
pytest tests/

# With coverage
pytest tests/ --cov=ragguard --cov-report=html

# Security tests only
pytest tests/ -m security

# Skip integration tests
pytest tests/ -m "not integration"
```

## Security Test Scripts

```bash
cd examples/arxiv-scale
python3 test_security_bypass.py           # Security attacks
python3 test_concurrent_users.py          # Concurrent load
python3 test_stability.py                 # Long-running stability
python3 test_multi_tenant.py              # Tenant isolation
python3 test_adversarial.py               # Malicious inputs
python3 test_institution_scoped_admin.py  # Institution-scoped admin
```

## The Security Problem RAGGuard Solves

| Approach | Unauthorized Docs Exposed | Security |
|----------|---------------------------|----------|
| No filtering | All unauthorized docs | Completely insecure |
| Post-filtering | Temporarily exposed (then filtered) | Leaks data during retrieval |
| **RAGGuard** | **Zero** | Never exposes unauthorized data |

The key insight: post-filtering retrieves unauthorized documents then filters them out. Even if users don't see them, the data was still exposed to the retrieval layer. That's a data leak.

RAGGuard filters **during** vector search, so unauthorized documents are never retrieved in the first place.
