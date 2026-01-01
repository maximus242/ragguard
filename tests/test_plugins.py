"""
Tests for RAGGuard plugin system.

Tests plugin base classes, registry, and extensibility.
"""

from typing import Any, Dict, Optional

import pytest

from ragguard.plugins.base import AttributeProvider, AuditSink, CacheBackend
from ragguard.plugins.registry import PluginRegistry

# Mock Implementations for Testing
# ---------------------------------

class MockAuditSink(AuditSink):
    """Mock audit sink implementation for testing."""

    def __init__(self, **config):
        self.config = config
        self.entries = []
        self.closed = False

    def write(self, entry: Dict[str, Any]) -> None:
        """Write audit entry."""
        if self.closed:
            raise RuntimeError("Sink is closed")
        self.entries.append(entry)

    def close(self) -> None:
        """Close sink."""
        self.closed = True


class MockCacheBackend(CacheBackend):
    """Mock cache backend implementation for testing."""

    def __init__(self, **config):
        self.config = config
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        self.cache[key] = value

    def delete(self, key: str) -> None:
        """Delete cached value."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()


class MockAttributeProvider(AttributeProvider):
    """Mock attribute provider implementation for testing."""

    def __init__(self, **config):
        self.config = config
        self.attributes = config.get("attributes", {})

    def get_attributes(self, user_id: str) -> Dict[str, Any]:
        """Get user attributes."""
        return self.attributes.get(user_id, {})


# Base Class Tests
# ----------------

class TestAuditSinkBase:
    """Test AuditSink base class."""

    def test_audit_sink_write(self):
        """Test audit sink write method."""
        sink = MockAuditSink()
        entry = {
            "timestamp": "2024-01-01T00:00:00Z",
            "user_id": "alice",
            "query": "[vector]",
            "results_returned": 5
        }

        sink.write(entry)
        assert len(sink.entries) == 1
        assert sink.entries[0] == entry

    def test_audit_sink_multiple_writes(self):
        """Test multiple audit entries."""
        sink = MockAuditSink()

        for i in range(5):
            sink.write({"id": i})

        assert len(sink.entries) == 5

    def test_audit_sink_close(self):
        """Test audit sink close method."""
        sink = MockAuditSink()
        sink.write({"test": "data"})

        sink.close()
        assert sink.closed is True

        # Writing after close should fail
        with pytest.raises(RuntimeError, match="Sink is closed"):
            sink.write({"another": "entry"})

    def test_audit_sink_with_config(self):
        """Test audit sink with configuration."""
        sink = MockAuditSink(url="https://example.com", api_key="secret")

        assert sink.config["url"] == "https://example.com"
        assert sink.config["api_key"] == "secret"


class TestCacheBackendBase:
    """Test CacheBackend base class."""

    def test_cache_get_miss(self):
        """Test cache miss."""
        cache = MockCacheBackend()
        result = cache.get("nonexistent")

        assert result is None
        assert cache.miss_count == 1
        assert cache.hit_count == 0

    def test_cache_set_and_get(self):
        """Test cache set and get."""
        cache = MockCacheBackend()

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        assert cache.hit_count == 1
        assert cache.miss_count == 0

    def test_cache_delete(self):
        """Test cache delete."""
        cache = MockCacheBackend()

        cache.set("key1", "value1")
        cache.delete("key1")
        result = cache.get("key1")

        assert result is None

    def test_cache_clear(self):
        """Test cache clear."""
        cache = MockCacheBackend()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_with_ttl(self):
        """Test cache with TTL parameter."""
        cache = MockCacheBackend()

        # TTL is accepted but not enforced in test implementation
        cache.set("key1", "value1", ttl=60)
        result = cache.get("key1")

        assert result == "value1"


class TestAttributeProviderBase:
    """Test AttributeProvider base class."""

    def test_get_attributes(self):
        """Test getting user attributes."""
        provider = MockAttributeProvider(
            attributes={
                "alice": {"department": "engineering", "level": 5},
                "bob": {"department": "sales", "level": 3}
            }
        )

        alice_attrs = provider.get_attributes("alice")
        assert alice_attrs["department"] == "engineering"
        assert alice_attrs["level"] == 5

        bob_attrs = provider.get_attributes("bob")
        assert bob_attrs["department"] == "sales"

    def test_get_attributes_not_found(self):
        """Test getting attributes for nonexistent entity."""
        provider = MockAttributeProvider(attributes={})

        attrs = provider.get_attributes("nonexistent")
        assert attrs == {}

    def test_enrich_user(self):
        """Test enriching user context."""
        provider = MockAttributeProvider(
            attributes={
                "alice": {"department": "engineering", "groups": ["dev", "ops"]}
            }
        )

        user = {"id": "alice", "name": "Alice"}
        enriched = provider.enrich_user(user)

        assert enriched["id"] == "alice"
        assert enriched["name"] == "Alice"
        assert enriched["department"] == "engineering"
        assert enriched["groups"] == ["dev", "ops"]


# Plugin Registry Tests
# ----------------------

class TestPluginRegistry:
    """Test PluginRegistry."""

    def test_register_audit_sink(self):
        """Test registering an audit sink."""
        registry = PluginRegistry()
        registry.register_audit_sink("test", MockAuditSink)

        assert "test" in registry.list_audit_sinks()

    def test_get_audit_sink(self):
        """Test getting an audit sink."""
        registry = PluginRegistry()
        registry.register_audit_sink("test", MockAuditSink)

        sink = registry.get_audit_sink("test", url="https://example.com")

        assert isinstance(sink, MockAuditSink)
        assert sink.config["url"] == "https://example.com"

    def test_get_audit_sink_not_found(self):
        """Test getting nonexistent audit sink."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="Audit sink 'nonexistent' not registered"):
            registry.get_audit_sink("nonexistent")

    def test_register_invalid_audit_sink(self):
        """Test registering invalid audit sink."""
        registry = PluginRegistry()

        class NotAnAuditSink:
            pass

        with pytest.raises(TypeError, match="must be a subclass of AuditSink"):
            registry.register_audit_sink("invalid", NotAnAuditSink)

    def test_register_cache_backend(self):
        """Test registering a cache backend."""
        registry = PluginRegistry()
        registry.register_cache_backend("test", MockCacheBackend)

        assert "test" in registry.list_cache_backends()

    def test_get_cache_backend(self):
        """Test getting a cache backend."""
        registry = PluginRegistry()
        registry.register_cache_backend("test", MockCacheBackend)

        cache = registry.get_cache_backend("test", host="localhost", port=6379)

        assert isinstance(cache, MockCacheBackend)
        assert cache.config["host"] == "localhost"
        assert cache.config["port"] == 6379

    def test_get_cache_backend_not_found(self):
        """Test getting nonexistent cache backend."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="Cache backend 'nonexistent' not registered"):
            registry.get_cache_backend("nonexistent")

    def test_register_invalid_cache_backend(self):
        """Test registering invalid cache backend."""
        registry = PluginRegistry()

        class NotACacheBackend:
            pass

        with pytest.raises(TypeError, match="must be a subclass of CacheBackend"):
            registry.register_cache_backend("invalid", NotACacheBackend)

    def test_register_attribute_provider(self):
        """Test registering an attribute provider."""
        registry = PluginRegistry()
        registry.register_attribute_provider("test", MockAttributeProvider)

        assert "test" in registry.list_attribute_providers()

    def test_get_attribute_provider(self):
        """Test getting an attribute provider."""
        registry = PluginRegistry()
        registry.register_attribute_provider("test", MockAttributeProvider)

        provider = registry.get_attribute_provider(
            "test",
            attributes={"alice": {"role": "admin"}}
        )

        assert isinstance(provider, MockAttributeProvider)
        attrs = provider.get_attributes("alice")
        assert attrs["role"] == "admin"

    def test_get_attribute_provider_not_found(self):
        """Test getting nonexistent attribute provider."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="Attribute provider 'nonexistent' not registered"):
            registry.get_attribute_provider("nonexistent")

    def test_list_all_plugins(self):
        """Test listing all registered plugins."""
        registry = PluginRegistry()

        registry.register_audit_sink("audit1", MockAuditSink)
        registry.register_audit_sink("audit2", MockAuditSink)
        registry.register_cache_backend("cache1", MockCacheBackend)
        registry.register_attribute_provider("attr1", MockAttributeProvider)

        assert len(registry.list_audit_sinks()) == 2
        assert len(registry.list_cache_backends()) == 1
        assert len(registry.list_attribute_providers()) == 1

    def test_replace_existing_plugin(self):
        """Test replacing an existing plugin."""
        registry = PluginRegistry()

        registry.register_audit_sink("test", MockAuditSink)

        # Register again with same name (should replace)
        registry.register_audit_sink("test", MockAuditSink)

        # Should still only have one
        assert len(registry.list_audit_sinks()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
