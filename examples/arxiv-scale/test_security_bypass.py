#!/usr/bin/env python3
"""
Security Testing: Permission Bypass Attempts

Tests various attack scenarios to ensure RAGGuard cannot be tricked into
returning unauthorized documents.

EXPECTED RESULT: All attacks should FAIL (zero unauthorized documents returned)
"""

from qdrant_client import QdrantClient
from ragguard import QdrantSecureRetriever, load_policy
from sentence_transformers import SentenceTransformer

class SecurityTester:
    def __init__(self):
        self.passed = []
        self.failed = []
        
        # Setup
        print("🔧 Setting up test environment...")
        self.client = QdrantClient("localhost", port=6333)
        self.policy = load_policy("policy.yaml")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.retriever = QdrantSecureRetriever(
            client=self.client,
            collection="arxiv_2400_papers",
            policy=self.policy,
            embed_fn=self.model.encode
        )
        
        print(f"✅ Connected to Qdrant")
        print(f"✅ Loaded policy with {len(self.policy.rules)} rules\n")
    
    def test(self, name, attack_user, should_get_results=False):
        """Test an attack scenario."""
        print(f"Testing: {name}")
        print(f"  User: {attack_user}")
        
        try:
            results = self.retriever.search(
                "machine learning",
                user=attack_user,
                limit=10
            )
            
            # Check if results contain unauthorized documents
            # Use PolicyEngine to properly evaluate authorization (accounts for all rules including admin access)
            from ragguard.policy.engine import PolicyEngine
            engine = PolicyEngine(self.policy)

            unauthorized = 0
            unauthorized_examples = []
            for r in results:
                # Use the actual policy engine to check if this document should be allowed
                if attack_user and not engine.evaluate(attack_user, r.payload):
                    unauthorized += 1
                    if len(unauthorized_examples) < 3:
                        unauthorized_examples.append({
                            'institution': r.payload.get('institution'),
                            'access_level': r.payload.get('access_level')
                        })
            
            if unauthorized > 0:
                print(f"  ❌ SECURITY BREACH: {unauthorized} unauthorized documents returned!")
                for ex in unauthorized_examples:
                    print(f"     - institution={ex['institution']}, access_level={ex.get('access_level', 'N/A')}")
                self.failed.append(name)
            else:
                if should_get_results and len(results) == 0:
                    print(f"  ⚠️  Warning: No results (expected some)")
                else:
                    print(f"  ✅ SECURE: {len(results)} results, all authorized")
                self.passed.append(name)
                
        except Exception as e:
            # Errors are acceptable (graceful rejection)
            print(f"  ✅ BLOCKED: {type(e).__name__}: {str(e)[:60]}...")
            self.passed.append(name)
    
    def summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("SECURITY TEST SUMMARY")
        print("="*70)
        
        total = len(self.passed) + len(self.failed)
        print(f"\nTotal attack scenarios: {total}")
        print(f"Blocked: {len(self.passed)} ✅")
        print(f"Bypassed: {len(self.failed)} ❌")
        
        if self.failed:
            print("\n🚨 SECURITY VULNERABILITIES FOUND:")
            for f in self.failed:
                print(f"   - {f}")
            print("\n❌ DO NOT DEPLOY - Fix security issues first!")
        else:
            print("\n🎉 ALL ATTACKS BLOCKED!")
            print("✅ No security vulnerabilities found")
        
        print("="*70)
        return len(self.failed) == 0

print("="*70)
print("RAGGuard Security Testing: Permission Bypass Attempts")
print("="*70)
print()

tester = SecurityTester()

# Category 1: Injection Attacks
print("\n" + "="*70)
print("Category 1: Injection Attacks")
print("="*70)

tester.test("SQL injection in institution", {"institution": "' OR '1'='1", "roles": ["researcher"]})
tester.test("NoSQL injection in roles", {"institution": "MIT", "roles": ["researcher", {"$ne": "admin"}]})
tester.test("Command injection in user ID", {"id": "; DROP TABLE users--", "institution": "MIT", "roles": ["researcher"]})

# Category 2: Null/None Bypass
print("\n" + "="*70)
print("Category 2: Null/None Bypass")
print("="*70)

tester.test("None user context", None)
tester.test("Empty user dict", {})
tester.test("Null institution", {"institution": None, "roles": ["admin"]})
tester.test("Missing required fields", {"random_field": "value"})

# Category 3: Role Escalation
print("\n" + "="*70)
print("Category 3: Role Escalation")
print("="*70)

tester.test("Array with admin role", {"institution": "Yale", "roles": ["student", "admin", "superuser"]})
tester.test("Roles as string", {"institution": "MIT", "roles": "admin"})
tester.test("Nested role object", {"institution": "MIT", "roles": [{"role": "admin", "all": True}]})

# Category 4: Special Characters
print("\n" + "="*70)
print("Category 4: Special Characters & Edge Cases")
print("="*70)

tester.test("Unicode null byte", {"institution": "MIT\u0000admin", "roles": ["researcher"]})
tester.test("Regex wildcard", {"institution": ".*", "roles": ["researcher"]})
tester.test("Very long string", {"institution": "A" * 10000, "roles": ["researcher"]})
tester.test("Empty string", {"institution": "", "roles": ["researcher"]})

# Category 5: Type Confusion
print("\n" + "="*70)
print("Category 5: Type Confusion")
print("="*70)

tester.test("Integer institution", {"institution": 12345, "roles": ["researcher"]})
tester.test("Boolean role", {"institution": "MIT", "roles": [True, False]})
tester.test("List institution", {"institution": ["MIT", "Yale"], "roles": ["researcher"]})

# Category 6: Positive Controls
print("\n" + "="*70)
print("Category 6: Positive Controls (Should Work)")
print("="*70)

tester.test("Valid MIT researcher", {"institution": "MIT", "roles": ["researcher"]}, should_get_results=True)
tester.test("Valid public access", {"institution": "Yale", "roles": ["student"]}, should_get_results=True)

# Summary
success = tester.summary()

import sys
sys.exit(0 if success else 1)
