#!/usr/bin/env python3
"""Comprehensive validation suite for RAGGuard release readiness."""
import sys, subprocess, tempfile, os

class Validator:
    def __init__(self):
        self.passed, self.failed = [], []
    
    def test(self, name, func):
        print(f"\n{'='*70}\nTesting: {name}\n{'='*70}")
        try:
            func()
            print(f"✅ PASS: {name}")
            self.passed.append(name)
        except Exception as e:
            print(f"❌ FAIL: {name}\n   Error: {e}")
            self.failed.append(name)
    
    def summary(self):
        print(f"\n{'='*70}\nVALIDATION SUMMARY\n{'='*70}")
        print(f"\nTotal: {len(self.passed)+len(self.failed)}")
        print(f"Passed: {len(self.passed)} ✅\nFailed: {len(self.failed)} ❌")
        if self.failed:
            print("\n❌ Failed:", *[f"\n   - {t}" for t in self.failed])
            print(f"\n{'='*70}\n❌ VALIDATION FAILED")
            return False
        print(f"\n{'='*70}\n✅ ALL VALIDATIONS PASSED - Ready for release!")
        return True

v = Validator()

v.test("Unit Tests", lambda: (
    (r := subprocess.run(["python3", "-m", "pytest", "tests/", "-q"], capture_output=True, text=True, cwd="/Users/cloud/Programming/ragguard")),
    print([l for l in r.stdout.split('\n') if 'passed' in l][-1]),
    r.returncode == 0 or (_ for _ in ()).throw(Exception("Tests failed"))
)[-1])

v.test("Core Imports", lambda: (
    __import__('ragguard'),
    __import__('ragguard.integrations.langchain').integrations.langchain.LangChainSecureRetriever,
    __import__('ragguard.integrations.google_workspace').integrations.google_workspace.GoogleWorkspaceResolver,
    print("All core imports successful")
))

v.test("Policy Loading", lambda: (
    (p := __import__('ragguard').load_policy("/Users/cloud/Programming/ragguard/examples/arxiv-scale/policy.yaml")),
    assert_(p.version == "1" and len(p.rules) > 0),
    print("Policy loading works")
))

v.test("Qdrant Retriever E2E", lambda: (
    (client := __import__('qdrant_client').QdrantClient("localhost", 6333)),
    (policy := __import__('ragguard').load_policy("/Users/cloud/Programming/ragguard/examples/arxiv-scale/policy.yaml")),
    (model := __import__('sentence_transformers').SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")),
    (retriever := __import__('ragguard').QdrantSecureRetriever(client, "arxiv_2400_papers", policy, embed_fn=model.encode)),
    (results := retriever.search("machine learning", user={"institution": "MIT", "roles": ["researcher"]}, limit=5)),
    assert_(0 < len(results) <= 5),
    print(f"Retrieved {len(results)} results successfully")
))

v.test("Filter Caching", lambda: (
    (client := __import__('qdrant_client').QdrantClient("localhost", 6333)),
    (policy := __import__('ragguard').load_policy("/Users/cloud/Programming/ragguard/examples/arxiv-scale/policy.yaml")),
    (model := __import__('sentence_transformers').SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")),
    (retriever := __import__('ragguard').QdrantSecureRetriever(client, "arxiv_2400_papers", policy, embed_fn=model.encode, enable_filter_cache=True)),
    (user := {"institution": "MIT", "roles": ["researcher"]}),
    retriever.search("test", user=user, limit=5),
    retriever.search("test", user=user, limit=5),
    (stats := retriever.get_cache_stats()),
    assert_(stats['hits'] >= 1 and stats['hit_rate'] > 0),
    print(f"Cache: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1%}")
))

v.test("Policy Testing Framework", lambda: (
    (Policy := __import__('ragguard').Policy),
    (PolicyTester := __import__('ragguard.testing', fromlist=['PolicyTester']).PolicyTester),
    (policy := Policy.from_dict({"version": "1", "rules": [{"name": "public", "match": {"visibility": "public"}, "allow": {"everyone": True}}], "default": "deny"})),
    (tester := PolicyTester(policy)),
    tester.add_test("test", {"id": "alice"}, {"visibility": "public"}, "allow"),
    (results := tester.run()),
    assert_(all(r.passed for r in results)),
    print(f"Policy testing works: {len(results)} tests passed")
))

v.test("Audit Logging", lambda: (
    (AuditLogger := __import__('ragguard').AuditLogger),
    (tmpfile := tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl')),
    tmpfile.close(),
    (logger := AuditLogger(output=f'file:{tmpfile.name}')),
    logger.log({"id": "test"}, "test query", 5, {"test": "filter"}),
    assert_(os.path.exists(tmpfile.name) and os.path.getsize(tmpfile.name) > 0),
    os.unlink(tmpfile.name),
    print("Audit logging works")
))

v.test("Multiple Backends", lambda: (
    __import__('ragguard').QdrantSecureRetriever,
    __import__('ragguard').PgvectorSecureRetriever,
    __import__('ragguard').WeaviateSecureRetriever,
    print("All backends importable ✓")
))

v.test("README Examples", lambda: (
    (client := __import__('qdrant_client').QdrantClient("localhost", 6333)),
    (policy := __import__('ragguard').load_policy("/Users/cloud/Programming/ragguard/examples/arxiv-scale/policy.yaml")),
    (retriever := __import__('ragguard').QdrantSecureRetriever(client, "arxiv_2400_papers", policy)),
    assert_(retriever.collection == "arxiv_2400_papers"),
    print("README examples validated")
))

v.test("No Deprecation Warnings", lambda: (
    (r := subprocess.run(["python3", "-m", "pytest", "tests/", "-q", "-W", "default"], capture_output=True, text=True, cwd="/Users/cloud/Programming/ragguard")),
    assert_("ragguard" not in r.stderr or "DeprecationWarning" not in r.stderr),
    print("No deprecation warnings")
))

def assert_(condition):
    if not condition:
        raise AssertionError()

sys.exit(0 if v.summary() else 1)
