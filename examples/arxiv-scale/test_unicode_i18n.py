#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode and Internationalization Tests for RAGGuard v0.2.0

Tests that RAGGuard handles non-ASCII characters correctly.
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine

print("=" * 80)
print("Unicode and Internationalization Tests")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test(name, func):
    """Run a test and track results."""
    global tests_passed, tests_failed
    print(f"\n🔍 {name}")
    try:
        result = func()
        if result:
            print(f"   ✅ PASS")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

# ============================================================================
# UNICODE TESTS
# ============================================================================

def test_chinese_characters():
    """Test Chinese characters in conditions"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "chinese",
            "allow": {
                "everyone": True,
                "conditions": ["document.title == '机器学习'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching Chinese text
    doc1 = {"title": "机器学习"}
    if not engine.evaluate({}, doc1):
        print(f"      Chinese character matching failed")
        return False

    # Test non-matching
    doc2 = {"title": "深度学习"}
    if engine.evaluate({}, doc2):
        print(f"      Chinese character non-match failed")
        return False

    print(f"      Chinese characters handled correctly")
    return True

test("Chinese characters", test_chinese_characters)

def test_accented_characters():
    """Test accented characters (é, ñ, ü)"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "accented",
            "allow": {
                "conditions": ["user.name == 'José'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching
    user1 = {"name": "José"}
    if not engine.evaluate(user1, {}):
        print(f"      Accented character matching failed")
        return False

    # Test non-matching
    user2 = {"name": "Jose"}  # No accent
    if engine.evaluate(user2, {}):
        print(f"      Accented vs non-accented should differ")
        return False

    print(f"      Accented characters handled correctly")
    return True

test("Accented characters (José, café, etc.)", test_accented_characters)

def test_emoji_in_conditions():
    """Test emoji in conditions"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "emoji",
            "allow": {
                "everyone": True,
                "conditions": ["document.reaction in ['👍', '❤️', '🚀']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching emoji
    doc1 = {"reaction": "👍"}
    if not engine.evaluate({}, doc1):
        print(f"      Emoji matching failed")
        return False

    # Test non-matching emoji
    doc2 = {"reaction": "💩"}
    if engine.evaluate({}, doc2):
        print(f"      Emoji non-match failed")
        return False

    print(f"      Emoji handled correctly")
    return True

test("Emoji in conditions", test_emoji_in_conditions)

def test_arabic_characters():
    """Test Arabic characters (RTL text)"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "arabic",
            "allow": {
                "everyone": True,
                "conditions": ["document.language == 'العربية'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching
    doc1 = {"language": "العربية"}
    if not engine.evaluate({}, doc1):
        print(f"      Arabic character matching failed")
        return False

    # Test non-matching
    doc2 = {"language": "English"}
    if engine.evaluate({}, doc2):
        print(f"      Arabic character non-match failed")
        return False

    print(f"      Arabic characters (RTL) handled correctly")
    return True

test("Arabic characters (RTL)", test_arabic_characters)

def test_cyrillic_characters():
    """Test Cyrillic characters"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "cyrillic",
            "allow": {
                "everyone": True,
                "conditions": ["document.city == 'Москва'"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching
    doc1 = {"city": "Москва"}
    if not engine.evaluate({}, doc1):
        print(f"      Cyrillic character matching failed")
        return False

    # Test non-matching
    doc2 = {"city": "Moscow"}
    if engine.evaluate({}, doc2):
        print(f"      Cyrillic vs Latin should differ")
        return False

    print(f"      Cyrillic characters handled correctly")
    return True

test("Cyrillic characters (Москва)", test_cyrillic_characters)

def test_unicode_in_lists():
    """Test Unicode characters in list literals"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "unicode-list",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['机器学习', 'AI', '深度学习']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test matching
    doc1 = {"category": "机器学习"}
    if not engine.evaluate({}, doc1):
        print(f"      Unicode in list matching failed")
        return False

    doc2 = {"category": "AI"}
    if not engine.evaluate({}, doc2):
        print(f"      ASCII in Unicode list matching failed")
        return False

    # Test non-matching
    doc3 = {"category": "Other"}
    if engine.evaluate({}, doc3):
        print(f"      Unicode list non-match failed")
        return False

    print(f"      Unicode in lists handled correctly")
    return True

test("Unicode in list literals", test_unicode_in_lists)

def test_mixed_scripts():
    """Test mixing different scripts in same policy"""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "mixed",
            "allow": {
                "conditions": [
                    "user.name == 'José'",
                    "document.category in ['AI', '机器学习', 'العربية']",
                    "document.city != 'Москва'"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test all conditions met
    user = {"name": "José"}
    doc = {"category": "AI", "city": "Paris"}
    if not engine.evaluate(user, doc):
        print(f"      Mixed scripts with all conditions failed")
        return False

    # Test one condition fails
    doc2 = {"category": "AI", "city": "Москва"}
    if engine.evaluate(user, doc2):
        print(f"      Mixed scripts with failing condition should deny")
        return False

    print(f"      Mixed scripts in same policy handled correctly")
    return True

test("Mixed scripts in same policy", test_mixed_scripts)

def test_unicode_byte_size_limit():
    """Test that Unicode characters count correctly in size limits"""
    # Create a large list with multi-byte Unicode characters
    # Each emoji is ~4 bytes in UTF-8
    # Create a list with many emoji to test byte size limits
    emoji_list = ", ".join(["'🚀'"] * 5000)  # 5000 emoji should exceed 100KB limit
    many_emoji = f"[{emoji_list}]"

    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "many-emoji",
                "allow": {
                    "everyone": True,
                    "conditions": [f"document.reaction in {many_emoji}"]
                }
            }],
            "default": "deny"
        })
        print(f"      ERROR: Should have rejected oversized list")
        return False
    except ValueError as e:
        if "too large" in str(e).lower() or "list literal too large" in str(e).lower():
            print(f"      Unicode correctly counted in byte size limits")
            return True
        else:
            print(f"      Unexpected error: {str(e)[:200]}")
            return False

test("Unicode byte size limits", test_unicode_byte_size_limit)

# Summary
print("\n" + "=" * 80)
print("UNICODE/I18N TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL UNICODE/I18N TESTS PASSED!")
    print("\n✅ RAGGuard correctly handles:")
    print("  - Chinese characters (机器学习)")
    print("  - Accented characters (José, café)")
    print("  - Emoji (👍, ❤️, 🚀)")
    print("  - Arabic characters (العربية)")
    print("  - Cyrillic characters (Москва)")
    print("  - Mixed scripts in same policy")
    print("  - Unicode in list literals")
    print("  - Unicode byte counting in size limits")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
