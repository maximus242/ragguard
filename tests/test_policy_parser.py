"""
Tests for policy parsing.
"""

import tempfile
from pathlib import Path

import pytest

from ragguard.exceptions import PolicyError, PolicyValidationError
from ragguard.policy import Policy, PolicyParser


def test_parse_valid_policy_from_dict():
    """Test parsing a valid policy from dictionary."""
    policy_dict = {
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True},
            }
        ],
        "default": "deny",
    }

    policy = PolicyParser.from_dict(policy_dict)

    assert policy.version == "1"
    assert len(policy.rules) == 1
    assert policy.rules[0].name == "public-docs"
    assert policy.default == "deny"


def test_parse_policy_from_file():
    """Test parsing a policy from YAML file."""
    yaml_content = """
version: "1"
rules:
  - name: "test-rule"
    allow:
      everyone: true
default: deny
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        policy = PolicyParser.from_file(temp_path)
        assert policy.version == "1"
        assert len(policy.rules) == 1
        assert policy.rules[0].name == "test-rule"
    finally:
        Path(temp_path).unlink()


def test_parse_policy_missing_version():
    """Test that missing version raises error."""
    policy_dict = {
        "rules": [
            {
                "name": "test",
                "allow": {"everyone": True},
            }
        ],
    }

    with pytest.raises(PolicyValidationError):
        PolicyParser.from_dict(policy_dict)


def test_parse_policy_unsupported_version():
    """Test that unsupported version raises error."""
    policy_dict = {
        "version": "99",
        "rules": [
            {
                "name": "test",
                "allow": {"everyone": True},
            }
        ],
    }

    with pytest.raises(PolicyValidationError, match="Unsupported policy version"):
        PolicyParser.from_dict(policy_dict)


def test_parse_policy_no_rules():
    """Test that policy with no rules raises error."""
    policy_dict = {
        "version": "1",
        "rules": [],
    }

    with pytest.raises(PolicyValidationError, match="at least one rule"):
        PolicyParser.from_dict(policy_dict)


def test_parse_policy_invalid_condition():
    """Test that invalid conditions raise error."""
    policy_dict = {
        "version": "1",
        "rules": [
            {
                "name": "test",
                "allow": {
                    "conditions": ["user.field ~= 10"]  # ~= not supported
                },
            }
        ],
    }

    with pytest.raises(PolicyValidationError):
        PolicyParser.from_dict(policy_dict)


def test_parse_policy_file_not_found():
    """Test that missing file raises error."""
    with pytest.raises(PolicyError, match="not found"):
        PolicyParser.from_file("/nonexistent/path.yaml")


def test_parse_policy_invalid_yaml():
    """Test that invalid YAML raises error."""
    invalid_yaml = """
version: "1"
rules:
  - name: "test"
    allow:
      everyone: true
    invalid_indent
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(invalid_yaml)
        temp_path = f.name

    try:
        with pytest.raises(PolicyError, match="Failed to parse YAML"):
            PolicyParser.from_file(temp_path)
    finally:
        Path(temp_path).unlink()


def test_parse_policy_with_all_features():
    """Test parsing a policy with all features."""
    policy_dict = {
        "version": "1",
        "rules": [
            {
                "name": "public",
                "match": {"visibility": "public"},
                "allow": {"everyone": True},
            },
            {
                "name": "department",
                "match": {"type": "internal"},
                "allow": {
                    "roles": ["manager", "employee"],
                    "conditions": ["user.department == document.department"],
                },
            },
            {
                "name": "shared",
                "allow": {
                    "conditions": ["user.id in document.shared_with"],
                },
            },
        ],
        "default": "deny",
    }

    policy = PolicyParser.from_dict(policy_dict)

    assert len(policy.rules) == 3
    assert policy.rules[0].allow.everyone is True
    assert policy.rules[1].allow.roles == ["manager", "employee"]
    assert policy.rules[1].match == {"type": "internal"}
    assert policy.rules[2].allow.conditions == ["user.id in document.shared_with"]
