"""
Tests for graph_base.py and plugins/base.py to improve coverage.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Graph Base Retriever Tests
# ============================================================================

class TestBaseGraphRetriever:
    """Tests for BaseGraphRetriever."""

    def create_mock_graph_retriever(self):
        """Create a concrete mock implementation for testing."""
        from ragguard import Policy
        from ragguard.retrievers.graph_base import BaseGraphRetriever

        class MockGraphRetriever(BaseGraphRetriever):
            @property
            def backend_name(self) -> str:
                return "mock_graph"

            def _build_filter(self, user: dict):
                return {"user": user.get("id")}

            def _execute_graph_query(self, query, permission_filter, limit, **kwargs):
                return [{"id": "result1"}, {"id": "result2"}]

            def _execute_property_search(self, properties, permission_filter, limit, **kwargs):
                return [{"id": "prop_result"}]

            def _build_permission_clause(self, user):
                return ("WHERE user.id = $user_id", {"user_id": user.get("id")})

            def _execute_traversal(self, start_node_id, relationship_type, user, direction, depth, limit, **kwargs):
                return [{"id": "traversed_node"}]

            def _check_backend_health(self):
                return {"status": "healthy"}

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "allow-all", "allow": {"conditions": ["document.public == True"]}}],
            "default": "allow"
        })

        mock_client = MagicMock()
        return MockGraphRetriever(
            client=mock_client,
            node_label="Document",
            policy=policy,
            database="test_db"
        )

    def test_init_graph_retriever(self):
        """Test graph retriever initialization."""
        retriever = self.create_mock_graph_retriever()

        assert retriever.node_label == "Document"
        assert retriever.database == "test_db"
        assert retriever.backend_name == "mock_graph"

    def test_execute_search_with_string_query(self):
        """Test _execute_search routes string queries to graph query."""
        retriever = self.create_mock_graph_retriever()

        # String query should route to _execute_graph_query
        results = retriever._execute_search(
            query="MATCH (n:Document) RETURN n",
            filter={"user": "alice"},
            limit=10
        )

        assert len(results) == 2
        assert results[0]["id"] == "result1"

    def test_execute_search_with_dict_query(self):
        """Test _execute_search routes dict queries to property search."""
        retriever = self.create_mock_graph_retriever()

        # Dict query should route to _execute_property_search
        results = retriever._execute_search(
            query={"status": "active"},
            filter={"user": "alice"},
            limit=10
        )

        assert len(results) == 1
        assert results[0]["id"] == "prop_result"

    def test_execute_search_with_list_query(self):
        """Test _execute_search routes list queries to vector search."""
        retriever = self.create_mock_graph_retriever()

        # List query should route to _execute_vector_search (NotImplemented by default)
        with pytest.raises(NotImplementedError) as exc:
            retriever._execute_search(
                query=[0.1, 0.2, 0.3],
                filter={"user": "alice"},
                limit=10
            )

        assert "does not support vector similarity search" in str(exc.value)

    def test_execute_search_with_invalid_type(self):
        """Test _execute_search raises error for unsupported types."""
        retriever = self.create_mock_graph_retriever()

        with pytest.raises(ValueError) as exc:
            retriever._execute_search(
                query=12345,  # Invalid type
                filter={"user": "alice"},
                limit=10
            )

        assert "Unsupported query type" in str(exc.value)
        assert "int" in str(exc.value)

    def test_traverse_valid_direction(self):
        """Test traverse with valid directions."""
        retriever = self.create_mock_graph_retriever()

        for direction in ["outgoing", "incoming", "both"]:
            results = retriever.traverse(
                start_node_id="node1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction=direction,
                depth=2,
                limit=50
            )

            assert len(results) == 1
            assert results[0]["id"] == "traversed_node"

    def test_traverse_invalid_direction(self):
        """Test traverse raises error for invalid direction."""
        retriever = self.create_mock_graph_retriever()

        with pytest.raises(ValueError) as exc:
            retriever.traverse(
                start_node_id="node1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="sideways",  # Invalid
                depth=1,
                limit=10
            )

        assert "Invalid direction" in str(exc.value)
        assert "sideways" in str(exc.value)


# ============================================================================
# Plugin Base Classes Tests
# ============================================================================

class TestAuditSinkBase:
    """Tests for AuditSink base class."""

    def test_audit_sink_implementation(self):
        """Test implementing AuditSink interface."""
        from ragguard.plugins.base import AuditSink

        entries = []

        class TestAuditSink(AuditSink):
            def write(self, entry: Dict[str, Any]) -> None:
                entries.append(entry)

        sink = TestAuditSink()
        sink.write({"timestamp": "2024-01-01", "user_id": "alice"})

        assert len(entries) == 1
        assert entries[0]["user_id"] == "alice"

    def test_audit_sink_close_default(self):
        """Test AuditSink.close() default implementation."""
        from ragguard.plugins.base import AuditSink

        class TestAuditSink(AuditSink):
            def write(self, entry: Dict[str, Any]) -> None:
                pass

        sink = TestAuditSink()
        # close() should not raise even without override
        sink.close()


class TestCacheBackendBase:
    """Tests for CacheBackend base class."""

    def test_cache_backend_implementation(self):
        """Test implementing CacheBackend interface."""
        from ragguard.plugins.base import CacheBackend

        cache = {}

        class TestCacheBackend(CacheBackend):
            def get(self, key: str):
                return cache.get(key)

            def set(self, key: str, value: Any, ttl: int = 300) -> None:
                cache[key] = value

            def delete(self, key: str) -> bool:
                if key in cache:
                    del cache[key]
                    return True
                return False

            def clear(self) -> None:
                cache.clear()

        backend = TestCacheBackend()
        backend.set("key1", "value1")
        assert backend.get("key1") == "value1"
        assert backend.delete("key1") is True
        assert backend.delete("key1") is False  # Already deleted

    def test_cache_backend_get_stats_default(self):
        """Test CacheBackend.get_stats() default implementation."""
        from ragguard.plugins.base import CacheBackend

        class TestCacheBackend(CacheBackend):
            def get(self, key: str):
                return None

            def set(self, key: str, value: Any, ttl: int = 300) -> None:
                pass

            def delete(self, key: str) -> bool:
                return False

            def clear(self) -> None:
                pass

        backend = TestCacheBackend()
        stats = backend.get_stats()
        assert stats == {}

    def test_cache_backend_close_default(self):
        """Test CacheBackend.close() default implementation."""
        from ragguard.plugins.base import CacheBackend

        class TestCacheBackend(CacheBackend):
            def get(self, key: str):
                return None

            def set(self, key: str, value: Any, ttl: int = 300) -> None:
                pass

            def delete(self, key: str) -> bool:
                return False

            def clear(self) -> None:
                pass

        backend = TestCacheBackend()
        backend.close()  # Should not raise


class TestAttributeProviderBase:
    """Tests for AttributeProvider base class."""

    def test_attribute_provider_implementation(self):
        """Test implementing AttributeProvider interface."""
        from ragguard.plugins.base import AttributeProvider

        class TestAttributeProvider(AttributeProvider):
            def get_attributes(self, user_id: str) -> Dict[str, Any]:
                if user_id == "alice":
                    return {"department": "engineering", "level": 5}
                return {}

        provider = TestAttributeProvider()
        attrs = provider.get_attributes("alice")
        assert attrs["department"] == "engineering"
        assert attrs["level"] == 5

    def test_attribute_provider_enrich_user(self):
        """Test AttributeProvider.enrich_user() method."""
        from ragguard.plugins.base import AttributeProvider

        class TestAttributeProvider(AttributeProvider):
            def get_attributes(self, user_id: str) -> Dict[str, Any]:
                return {"department": "engineering", "groups": ["dev", "ops"]}

        provider = TestAttributeProvider()

        # User with ID gets enriched
        user = {"id": "alice", "name": "Alice"}
        enriched = provider.enrich_user(user)

        assert enriched["id"] == "alice"
        assert enriched["name"] == "Alice"
        assert enriched["department"] == "engineering"
        assert enriched["groups"] == ["dev", "ops"]

    def test_attribute_provider_enrich_user_no_id(self):
        """Test AttributeProvider.enrich_user() with no user ID."""
        from ragguard.plugins.base import AttributeProvider

        class TestAttributeProvider(AttributeProvider):
            def get_attributes(self, user_id: str) -> Dict[str, Any]:
                raise RuntimeError("Should not be called")

        provider = TestAttributeProvider()

        # User without ID returns unchanged
        user = {"name": "Anonymous"}
        enriched = provider.enrich_user(user)

        assert enriched == user  # Unchanged

    def test_attribute_provider_close_default(self):
        """Test AttributeProvider.close() default implementation."""
        from ragguard.plugins.base import AttributeProvider

        class TestAttributeProvider(AttributeProvider):
            def get_attributes(self, user_id: str) -> Dict[str, Any]:
                return {}

        provider = TestAttributeProvider()
        provider.close()  # Should not raise


# ============================================================================
# Policy Coverage Tester Tests
# ============================================================================

class TestPolicyCoverageTester:
    """Tests for PolicyCoverageTester."""

    def test_generate_role_coverage_tests(self):
        """Test generating role-based coverage tests."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "role-check", "allow": {"roles": ["admin", "user", "guest"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_role_coverage_tests(
            roles=["admin", "user", "guest"],
            sample_document={"id": "doc1"}
        )

        assert len(tester.test_cases) == 3

    def test_generate_role_coverage_tests_with_template(self):
        """Test generating role coverage with user template."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "role-check", "allow": {"roles": ["admin", "user"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_role_coverage_tests(
            roles=["admin", "user"],
            sample_document={"id": "doc1"},
            user_template={"id": "test_user", "department": "engineering"}
        )

        assert len(tester.test_cases) == 2

    def test_generate_field_coverage_tests(self):
        """Test generating field-based coverage tests."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "status-check", "allow": {"conditions": ["document.status == 'active'"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_field_coverage_tests(
            field="status",
            values=["active", "pending", "closed"],
            user={"id": "alice"}
        )

        assert len(tester.test_cases) == 3

    def test_generate_field_coverage_tests_with_template(self):
        """Test generating field coverage with document template."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "priority-check", "allow": {"conditions": ["document.priority == 'high'"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_field_coverage_tests(
            field="priority",
            values=["high", "medium", "low"],
            user={"id": "alice"},
            document_template={"id": "doc1", "type": "task"}
        )

        assert len(tester.test_cases) == 3

    def test_generate_department_coverage_tests(self):
        """Test generating department-based coverage tests."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "dept-match", "allow": {"conditions": ["user.department == document.department"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_department_coverage_tests(
            departments=["engineering", "sales", "hr"]
        )

        # 2 tests per department (same + cross)
        assert len(tester.test_cases) == 6

    def test_generate_department_coverage_tests_with_sample(self):
        """Test generating department coverage with sample document."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "dept-match", "allow": {"conditions": ["user.department == document.department"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_department_coverage_tests(
            departments=["engineering", "sales"],
            sample_document={"id": "custom_doc", "type": "report"}
        )

        assert len(tester.test_cases) == 4

    def test_print_results(self):
        """Test print_results method."""
        from ragguard import Policy
        try:
            from ragguard_enterprise.testing.coverage import PolicyCoverageTester
        except ImportError:
            pytest.skip("ragguard_enterprise required for this test")

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "role-check", "allow": {"roles": ["admin"]}}],
            "default": "deny"
        })

        tester = PolicyCoverageTester(policy)
        tester.generate_role_coverage_tests(
            roles=["admin"],
            sample_document={"id": "doc1"}
        )

        # Should not raise
        tester.print_results(verbose=True)
