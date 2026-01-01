"""
Tests for RAGGuard CLI tool.

Tests command-line interface for policy validation and testing.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ragguard.cli import cmd_explain, cmd_filters, cmd_show, cmd_test, cmd_validate, load_json, main


class TestLoadJSON:
    """Test JSON loading from files and strings."""

    def test_load_json_from_file(self):
        """Test loading JSON from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_file = f.name

        try:
            data = load_json(temp_file)
            assert data == {"test": "data"}
        finally:
            os.unlink(temp_file)

    def test_load_json_from_string(self):
        """Test loading JSON from string."""
        json_str = '{"test": "data"}'
        data = load_json(json_str)
        assert data == {"test": "data"}

    def test_load_json_invalid(self):
        """Test loading invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json("not-a-file-or-json")


class TestValidateCommand:
    """Test policy validation command."""

    def test_validate_valid_policy(self, capsys):
        """Test validating a valid policy."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: admin_access
    allow:
      roles: [admin]
default: deny
""")
            temp_file = f.name

        try:
            # Create args mock
            class Args:
                policy = temp_file

            result = cmd_validate(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "Policy is valid." in captured.out
            assert "Version: 1" in captured.out
            assert "Rules: 1" in captured.out
        finally:
            os.unlink(temp_file)

    def test_validate_invalid_policy(self, capsys):
        """Test validating an invalid policy."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
invalid: yaml: content:
""")
            temp_file = f.name

        try:
            class Args:
                policy = temp_file

            result = cmd_validate(Args())
            assert result == 1

            captured = capsys.readouterr()
            assert "Policy validation failed" in captured.out
        finally:
            os.unlink(temp_file)


class TestTestCommand:
    """Test policy testing command."""

    def test_test_policy_pass(self, capsys):
        """Test policy test that passes."""
        # Create policy file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: public_access
    match:
      visibility: public
    allow:
      everyone: true
default: deny
""")
            policy_file = f.name

        # Create user file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"id": "alice"}, f)
            user_file = f.name

        # Create document file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"id": "doc1", "visibility": "public"}, f)
            doc_file = f.name

        try:
            class Args:
                policy = policy_file
                user = user_file
                document = doc_file

            result = cmd_test(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "ACCESS GRANTED" in captured.out
        finally:
            os.unlink(policy_file)
            os.unlink(user_file)
            os.unlink(doc_file)

    def test_test_policy_deny(self, capsys):
        """Test policy test that denies access."""
        # Create policy file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: admin_only
    allow:
      roles: [admin]
default: deny
""")
            policy_file = f.name

        # Create user file (non-admin user)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"id": "alice", "roles": ["user"]}, f)
            user_file = f.name

        # Create document file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"id": "doc1"}, f)
            doc_file = f.name

        try:
            class Args:
                policy = policy_file
                user = user_file
                document = doc_file

            result = cmd_test(Args())
            assert result == 0  # Command succeeds, just denies access

            captured = capsys.readouterr()
            assert "ACCESS DENIED" in captured.out
        finally:
            os.unlink(policy_file)
            os.unlink(user_file)
            os.unlink(doc_file)


class TestExplainCommand:
    """Test policy explanation command."""

    def test_explain_allow(self, capsys):
        """Test explaining an allow decision."""
        # Create policy file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: admin_access
    allow:
      roles: [admin]
default: deny
""")
            policy_file = f.name

        try:
            class Args:
                policy = policy_file
                user = '{"id": "alice", "roles": ["admin"]}'
                document = '{"id": "doc1"}'

            result = cmd_explain(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "ACCESS GRANTED" in captured.out
        finally:
            os.unlink(policy_file)

    def test_explain_deny(self, capsys):
        """Test explaining a deny decision."""
        # Create policy file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: admin_only
    allow:
      roles: [admin]
default: deny
""")
            policy_file = f.name

        try:
            class Args:
                policy = policy_file
                user = '{"id": "alice", "roles": ["user"]}'
                document = '{"id": "doc1"}'

            result = cmd_explain(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "ACCESS DENIED" in captured.out
        finally:
            os.unlink(policy_file)


class TestShowCommand:
    """Test policy show command."""

    def test_show_policy(self, capsys):
        """Test showing policy details."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: public_docs
    match:
      visibility: public
    allow:
      everyone: true
  - name: admin_access
    allow:
      roles: [admin, superuser]
  - name: department_access
    allow:
      conditions:
        - user.department == document.department
default: deny
""")
            temp_file = f.name

        try:
            class Args:
                policy = temp_file

            result = cmd_show(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "Version: 1" in captured.out
            assert "Default: deny" in captured.out
            assert "public_docs" in captured.out
            assert "admin_access" in captured.out
            assert "department_access" in captured.out
            assert "Everyone" in captured.out
            assert "admin" in captured.out
        finally:
            os.unlink(temp_file)

    def test_show_invalid_policy(self, capsys):
        """Test showing invalid policy."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("not: valid: yaml: content:")
            temp_file = f.name

        try:
            class Args:
                policy = temp_file

            result = cmd_show(Args())
            assert result == 1

            captured = capsys.readouterr()
            assert "Failed to show policy" in captured.out
        finally:
            os.unlink(temp_file)


class TestFiltersCommand:
    """Test filter generation command."""

    def test_filters_single_backend(self, capsys):
        """Test generating filters for a single backend."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: department_access
    allow:
      conditions:
        - user.department == document.department
default: deny
""")
            policy_file = f.name

        try:
            class Args:
                policy = policy_file
                user = '{"id": "alice", "department": "engineering"}'
                backend = "qdrant"

            result = cmd_filters(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "QDRANT" in captured.out
            assert "engineering" in captured.out
        finally:
            os.unlink(policy_file)

    def test_filters_all_backends(self, capsys):
        """Test generating filters for all backends."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: public_access
    match:
      visibility: public
    allow:
      everyone: true
default: deny
""")
            policy_file = f.name

        try:
            class Args:
                policy = policy_file
                user = '{"id": "alice"}'
                backend = None  # All backends

            result = cmd_filters(Args())
            assert result == 0

            captured = capsys.readouterr()
            # Should show multiple backends
            assert "QDRANT" in captured.out
            assert "CHROMADB" in captured.out
            assert "PGVECTOR" in captured.out
        finally:
            os.unlink(policy_file)

    def test_filters_from_file(self, capsys):
        """Test generating filters with user from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
version: "1"
rules:
  - name: role_access
    allow:
      roles: [admin]
default: deny
""")
            policy_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"id": "bob", "roles": ["admin"]}, f)
            user_file = f.name

        try:
            class Args:
                policy = policy_file
                user = user_file
                backend = "chromadb"

            result = cmd_filters(Args())
            assert result == 0

            captured = capsys.readouterr()
            assert "CHROMADB" in captured.out
        finally:
            os.unlink(policy_file)
            os.unlink(user_file)

    def test_filters_invalid_policy(self, capsys):
        """Test filter generation with invalid policy."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            temp_file = f.name

        try:
            class Args:
                policy = temp_file
                user = '{"id": "alice"}'
                backend = None

            result = cmd_filters(Args())
            assert result == 1

            captured = capsys.readouterr()
            assert "Filter generation failed" in captured.out
        finally:
            os.unlink(temp_file)


class TestMainFunction:
    """Test main CLI entry point."""

    def test_no_command(self, capsys):
        """Test running with no command shows help."""
        import sys
        old_argv = sys.argv
        sys.argv = ['ragguard']

        try:
            result = main()
            assert result == 1

            captured = capsys.readouterr()
            assert "usage:" in captured.out or "RAGGuard" in captured.out
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
