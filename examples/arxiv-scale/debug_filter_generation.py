#!/usr/bin/env python3
"""
Debug filter generation to find why unauthorized documents are returned.
"""

from ragguard import load_policy
from ragguard.filters.builder import to_qdrant_filter
import json

# Load the policy
policy = load_policy("policy.yaml")

# Create a MIT researcher user
mit_user = {"institution": "MIT", "roles": ["researcher"]}

print("=" * 70)
print("Filter Generation Debug")
print("=" * 70)

print(f"\n📋 Policy: {len(policy.rules)} rules")
for i, rule in enumerate(policy.rules):
    print(f"\nRule {i+1}: {rule.name}")
    print(f"  Match: {rule.match}")
    print(f"  Allow: roles={rule.allow.roles}, everyone={rule.allow.everyone}, conditions={rule.allow.conditions}")

print(f"\n👤 User: {mit_user}")

# Generate filter
print("\n🔍 Generating Qdrant filter...")
filter_obj = to_qdrant_filter(policy, mit_user)

print("\n📊 Generated Filter:")
print(f"Type: {type(filter_obj)}")

# Pretty print the filter structure
def print_filter(obj, indent=0):
    prefix = "  " * indent
    if hasattr(obj, '__dict__'):
        print(f"{prefix}{obj.__class__.__name__}(")
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            print(f"{prefix}  {key}=", end="")
            if isinstance(value, list):
                if value:
                    print("[")
                    for item in value:
                        print_filter(item, indent + 2)
                    print(f"{prefix}  ]")
                else:
                    print("[]")
            elif hasattr(value, '__dict__'):
                print()
                print_filter(value, indent + 2)
            else:
                print(f"{repr(value)}")
        print(f"{prefix})")
    else:
        print(f"{prefix}{repr(obj)}")

print_filter(filter_obj)

print("\n\n💡 Analysis:")
print("-" * 70)
print("\nFor MIT researcher, the filter should allow:")
print("  1. Documents where institution = 'MIT' (institution-access rule)")
print("  2. Documents where access_level = 'public' (public-access rule)")
print("  3. Documents where user.institution in document.institutions (admin-access rule condition)")
print("\nFor MIT researcher, the filter should DENY:")
print("  ❌ Documents where institution = 'Cornell' (not MIT)")
print("  ❌ Documents where institution = 'Harvard' (not MIT)")
print("  ❌ Documents where access_level = 'restricted' (not public)")

print("\n\n🔎 Let's check what the filter actually allows:")

# Manually inspect the filter conditions
from qdrant_client import models

def analyze_filter(obj, depth=0):
    indent = "  " * depth

    if isinstance(obj, models.Filter):
        if hasattr(obj, 'should') and obj.should:
            print(f"{indent}OR clause with {len(obj.should)} options:")
            for i, should_filter in enumerate(obj.should):
                print(f"{indent}  Option {i+1}:")
                analyze_filter(should_filter, depth + 2)
        elif hasattr(obj, 'must') and obj.must:
            print(f"{indent}AND clause with {len(obj.must)} conditions:")
            for i, must_filter in enumerate(obj.must):
                print(f"{indent}  Condition {i+1}:")
                analyze_filter(must_filter, depth + 2)
    elif isinstance(obj, models.FieldCondition):
        field = obj.key
        if hasattr(obj.match, 'value'):
            value = obj.match.value
            print(f"{indent}Field '{field}' must equal '{value}'")
        elif hasattr(obj.match, 'any'):
            values = obj.match.any
            print(f"{indent}Field '{field}' must be in {values}")
    else:
        print(f"{indent}{type(obj).__name__}: {obj}")

print("\n📋 Filter Structure:")
analyze_filter(filter_obj)

print("\n\n🚨 Expected Behavior:")
print("  The filter should use OR between rules:")
print("    (institution=MIT) OR (access_level=public) OR (MIT in institutions) OR ...")
print("\n  This means a document matches if ANY rule grants access.")
print("  Cornell papers should NOT match any of these conditions for a MIT user!")
