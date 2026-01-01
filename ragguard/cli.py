#!/usr/bin/env python3
"""RAGGuard CLI tool for policy validation and testing."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ragguard.policy import PolicyEngine, load_policy
from ragguard.policy.errors import PolicyErrorFormatter


def load_json(json_str_or_file: str) -> Dict[str, Any]:
    """Load JSON from string or file."""
    # Try as file first
    if Path(json_str_or_file).exists():
        with open(json_str_or_file) as f:
            return json.load(f)

    # Try as JSON string
    try:
        return json.loads(json_str_or_file)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {json_str_or_file}")


def cmd_validate(args):
    """Validate a policy file."""
    print(f"Validating policy: {args.policy}")
    print("=" * 60)

    try:
        policy = load_policy(args.policy)
        print("Policy is valid.")
        print()
        print(f"Version: {policy.version}")
        print(f"Rules: {len(policy.rules)}")
        print(f"Default: {policy.default}")
        print()

        # List rules
        print("Rules:")
        for i, rule in enumerate(policy.rules, 1):
            print(f"  {i}. {rule.name}")
            if rule.match:
                print(f"     Match: {list(rule.match.keys())}")
            if rule.allow.conditions:
                print(f"     Conditions: {len(rule.allow.conditions)}")
            if rule.allow.roles:
                print(f"     Roles: {rule.allow.roles}")

        return 0

    except Exception as e:
        print("Policy validation failed:")
        print()
        print(str(e))
        return 1


def cmd_test(args):
    """Test a policy against a user and document."""
    print(f"Testing policy: {args.policy}")
    print("=" * 60)
    print()

    try:
        # Load policy
        policy = load_policy(args.policy)

        # Load user and document
        user = load_json(args.user)
        document = load_json(args.document)

        # Create engine
        engine = PolicyEngine(policy)

        # Evaluate
        result = engine.evaluate(user, document)

        # Show results
        print("User:")
        print(json.dumps(user, indent=2))
        print()

        print("Document:")
        print(json.dumps(document, indent=2))
        print()

        if result:
            print("ACCESS GRANTED")
        else:
            print("ACCESS DENIED")

        return 0

    except Exception as e:
        print("Test failed:")
        print()
        print(str(e))
        return 1


def cmd_explain(args):
    """Explain why access was granted or denied."""
    print(f"Explaining policy: {args.policy}")
    print("=" * 60)
    print()

    try:
        # Load policy
        policy = load_policy(args.policy)

        # Load user and document
        user = load_json(args.user)
        document = load_json(args.document)

        # Create engine
        engine = PolicyEngine(policy)

        # Check if explain method exists
        if not hasattr(engine, 'explain'):
            print("Note: PolicyEngine.explain() not implemented yet")
            print("    Using basic evaluation...")
            result = engine.evaluate(user, document)

            print("User:")
            print(json.dumps(user, indent=2))
            print()

            print("Document:")
            print(json.dumps(document, indent=2))
            print()

            if result:
                print("ACCESS GRANTED")
                print()
                print("To see which rule matched, check the policy manually.")
            else:
                print("ACCESS DENIED")
                print()
                print("No rules matched. Default policy applied.")

            return 0

        # Use explain method
        explanation = engine.explain(user, document)

        print("User:")
        print(json.dumps(user, indent=2))
        print()

        print("Document:")
        print(json.dumps(document, indent=2))
        print()

        print("Explanation:")
        print(json.dumps(explanation, indent=2))

        return 0

    except Exception as e:
        print("Explain failed:")
        print()
        print(str(e))
        return 1


def cmd_filters(args):
    """Generate and show database filters for a user context."""
    print(f"Generating filters: {args.policy}")
    print("=" * 60)
    print()

    try:
        # Load policy and user
        policy = load_policy(args.policy)
        user = load_json(args.user)

        # Create engine
        engine = PolicyEngine(policy)

        # Determine which backends to show
        if args.backend:
            backends = [args.backend]
        else:
            backends = ["qdrant", "chromadb", "pinecone", "weaviate", "pgvector",
                       "milvus", "elasticsearch", "azure_search"]

        print("User context:")
        print(json.dumps(user, indent=2))
        print()

        for backend in backends:
            try:
                db_filter = engine.to_filter(user, backend=backend)

                print(f"── {backend.upper()} ──")
                if db_filter is None:
                    print("  (no filter - user has access to all documents)")
                elif isinstance(db_filter, str):
                    # SQL or expression-based filters
                    print(f"  {db_filter}")
                else:
                    # JSON-based filters
                    print(json.dumps(db_filter, indent=2, default=str))
                print()
            except Exception as e:
                print(f"── {backend.upper()} ──")
                print(f"  Error: {e}")
                print()

        return 0

    except Exception as e:
        print("Filter generation failed:")
        print()
        print(str(e))
        return 1


def cmd_show(args):
    """Show policy details in a readable format."""
    print(f"Policy: {args.policy}")
    print("=" * 60)
    print()

    try:
        policy = load_policy(args.policy)

        print(f"Version: {policy.version}")
        print(f"Default: {policy.default}")
        print()

        print("Rules:")
        print()

        for i, rule in enumerate(policy.rules, 1):
            print(f"{i}. {rule.name}")
            print("   " + "-" * 50)

            if rule.match:
                print("   Match:")
                for key, value in rule.match.items():
                    print(f"     • {key}: {value}")

            print("   Allow:")

            if rule.allow.everyone:
                print("     • Everyone")

            if rule.allow.roles:
                print(f"     • Roles: {', '.join(rule.allow.roles)}")

            if rule.allow.conditions:
                print("     • Conditions:")
                for cond in rule.allow.conditions:
                    print(f"       - {cond}")

            print()

        return 0

    except Exception as e:
        print("Failed to show policy:")
        print()
        print(str(e))
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='ragguard',
        description='RAGGuard policy validation and testing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Validate a policy file
  ragguard validate policy.yaml

  # Test a policy with user and document
  ragguard test policy.yaml --user '{"role": "admin"}' --document '{"status": "draft"}'

  # Test with JSON files
  ragguard test policy.yaml --user user.json --document doc.json

  # Explain why access was granted/denied
  ragguard explain policy.yaml --user user.json --document doc.json

  # Show policy details
  ragguard show policy.yaml

  # Generate database filters (dry-run, no database needed)
  ragguard filters policy.yaml --user '{"department": "finance"}'
  ragguard filters policy.yaml --user user.json --backend chromadb
        '''
    )

    # Add no-color flag
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a policy file'
    )
    validate_parser.add_argument(
        'policy',
        help='Path to policy YAML file'
    )

    # Test command
    test_parser = subparsers.add_parser(
        'test',
        help='Test a policy against a user and document'
    )
    test_parser.add_argument(
        'policy',
        help='Path to policy YAML file'
    )
    test_parser.add_argument(
        '--user',
        required=True,
        help='User context as JSON string or file path'
    )
    test_parser.add_argument(
        '--document',
        required=True,
        help='Document metadata as JSON string or file path'
    )

    # Explain command
    explain_parser = subparsers.add_parser(
        'explain',
        help='Explain why access was granted or denied'
    )
    explain_parser.add_argument(
        'policy',
        help='Path to policy YAML file'
    )
    explain_parser.add_argument(
        '--user',
        required=True,
        help='User context as JSON string or file path'
    )
    explain_parser.add_argument(
        '--document',
        required=True,
        help='Document metadata as JSON string or file path'
    )

    # Show command
    show_parser = subparsers.add_parser(
        'show',
        help='Show policy details in a readable format'
    )
    show_parser.add_argument(
        'policy',
        help='Path to policy YAML file'
    )

    # Filters command (dry-run filter generation)
    filters_parser = subparsers.add_parser(
        'filters',
        help='Generate database filters for a user (dry-run, no database needed)'
    )
    filters_parser.add_argument(
        'policy',
        help='Path to policy YAML file'
    )
    filters_parser.add_argument(
        '--user',
        required=True,
        help='User context as JSON string or file path'
    )
    filters_parser.add_argument(
        '--backend',
        choices=['qdrant', 'chromadb', 'pinecone', 'weaviate', 'pgvector',
                 'milvus', 'elasticsearch', 'azure_search'],
        help='Show filter for specific backend only (default: all backends)'
    )

    args = parser.parse_args()

    # Disable colors if requested or if not in TTY
    if args.no_color or not sys.stdout.isatty():
        PolicyErrorFormatter.disable_colors()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handlers
    if args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'test':
        return cmd_test(args)
    elif args.command == 'explain':
        return cmd_explain(args)
    elif args.command == 'show':
        return cmd_show(args)
    elif args.command == 'filters':
        return cmd_filters(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
