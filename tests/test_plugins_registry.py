"""
Comprehensive tests for plugins registry to maximize coverage.
"""

from unittest.mock import MagicMock

import pytest


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def test_creation(self):
        """Test PluginRegistry creation."""
        from ragguard.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        assert registry._audit_sinks == {}
        assert registry._cache_backends == {}
        assert registry._attribute_providers == {}


class TestAuditSinkRegistry:
    """Tests for audit sink registration."""

    def test_register_audit_sink(self):
        """Test registering an audit sink."""
        from ragguard.plugins.base import AuditSink
        from ragguard.plugins.registry import PluginRegistry

        class MyAuditSink(AuditSink):
            def write(self, entry): pass

        registry = PluginRegistry()
        registry.register_audit_sink("my_sink", MyAuditSink)

        assert "my_sink" in registry.list_audit_sinks()

    def test_register_audit_sink_invalid_type(self):
        """Test registering non-AuditSink class raises error."""
        from ragguard.plugins.registry import PluginRegistry

        class NotASink:
            pass

        registry = PluginRegistry()

        with pytest.raises(TypeError, match="must be a subclass of AuditSink"):
            registry.register_audit_sink("bad", NotASink)

    def test_get_audit_sink(self):
        """Test getting a registered audit sink."""
        from ragguard.plugins.base import AuditSink
        from ragguard.plugins.registry import PluginRegistry

        class TestSink(AuditSink):
            def __init__(self, url):
                self.url = url
            def write(self, entry): pass

        registry = PluginRegistry()
        registry.register_audit_sink("test", TestSink)

        sink = registry.get_audit_sink("test", url="http://example.com")

        assert sink.url == "http://example.com"

    def test_get_audit_sink_not_found(self):
        """Test getting non-existent audit sink."""
        from ragguard.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_audit_sink("unknown")


class TestCacheBackendRegistry:
    """Tests for cache backend registration."""

    def test_register_cache_backend(self):
        """Test registering a cache backend."""
        from ragguard.plugins.base import CacheBackend
        from ragguard.plugins.registry import PluginRegistry

        class MyCache(CacheBackend):
            def get(self, key): pass
            def set(self, key, value, ttl=None): pass
            def delete(self, key): pass
            def clear(self): pass

        registry = PluginRegistry()
        registry.register_cache_backend("my_cache", MyCache)

        assert "my_cache" in registry.list_cache_backends()

    def test_register_cache_backend_invalid_type(self):
        """Test registering non-CacheBackend class raises error."""
        from ragguard.plugins.registry import PluginRegistry

        class NotACache:
            pass

        registry = PluginRegistry()

        with pytest.raises(TypeError, match="must be a subclass of CacheBackend"):
            registry.register_cache_backend("bad", NotACache)

    def test_get_cache_backend(self):
        """Test getting a registered cache backend."""
        from ragguard.plugins.base import CacheBackend
        from ragguard.plugins.registry import PluginRegistry

        class TestCache(CacheBackend):
            def __init__(self, host):
                self.host = host
            def get(self, key): pass
            def set(self, key, value, ttl=None): pass
            def delete(self, key): pass
            def clear(self): pass

        registry = PluginRegistry()
        registry.register_cache_backend("test", TestCache)

        cache = registry.get_cache_backend("test", host="localhost")

        assert cache.host == "localhost"

    def test_get_cache_backend_not_found(self):
        """Test getting non-existent cache backend."""
        from ragguard.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_cache_backend("unknown")


class TestAttributeProviderRegistry:
    """Tests for attribute provider registration."""

    def test_register_attribute_provider(self):
        """Test registering an attribute provider."""
        from ragguard.plugins.base import AttributeProvider
        from ragguard.plugins.registry import PluginRegistry

        class MyProvider(AttributeProvider):
            def get_attributes(self, user_id): return {}

        registry = PluginRegistry()
        registry.register_attribute_provider("my_provider", MyProvider)

        assert "my_provider" in registry.list_attribute_providers()

    def test_register_attribute_provider_invalid_type(self):
        """Test registering non-AttributeProvider class raises error."""
        from ragguard.plugins.registry import PluginRegistry

        class NotAProvider:
            pass

        registry = PluginRegistry()

        with pytest.raises(TypeError, match="must be a subclass of AttributeProvider"):
            registry.register_attribute_provider("bad", NotAProvider)

    def test_get_attribute_provider(self):
        """Test getting a registered attribute provider."""
        from ragguard.plugins.base import AttributeProvider
        from ragguard.plugins.registry import PluginRegistry

        class TestProvider(AttributeProvider):
            def __init__(self, server):
                self.server = server
            def get_attributes(self, user_id): return {}

        registry = PluginRegistry()
        registry.register_attribute_provider("test", TestProvider)

        provider = registry.get_attribute_provider("test", server="ldap://example.com")

        assert provider.server == "ldap://example.com"

    def test_get_attribute_provider_not_found(self):
        """Test getting non-existent attribute provider."""
        from ragguard.plugins.registry import PluginRegistry

        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_attribute_provider("unknown")


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_global_register_audit_sink(self):
        """Test global register_audit_sink function."""
        from ragguard.plugins.base import AuditSink
        from ragguard.plugins.registry import (
            _registry,
            get_audit_sink,
            list_audit_sinks,
            register_audit_sink,
        )

        class GlobalTestSink(AuditSink):
            def __init__(self, sink_name="default"):
                self.sink_name = sink_name
            def write(self, entry): pass

        register_audit_sink("global_test_sink", GlobalTestSink)

        assert "global_test_sink" in list_audit_sinks()
        sink = get_audit_sink("global_test_sink", sink_name="custom")
        assert sink.sink_name == "custom"

    def test_global_register_cache_backend(self):
        """Test global register_cache_backend function."""
        from ragguard.plugins.base import CacheBackend
        from ragguard.plugins.registry import (
            get_cache_backend,
            list_cache_backends,
            register_cache_backend,
        )

        class GlobalTestCache(CacheBackend):
            def __init__(self, host="localhost"):
                self.host = host
            def get(self, key): pass
            def set(self, key, value, ttl=None): pass
            def delete(self, key): pass
            def clear(self): pass

        register_cache_backend("global_test_cache", GlobalTestCache)

        assert "global_test_cache" in list_cache_backends()
        cache = get_cache_backend("global_test_cache", host="redis://server")
        assert cache.host == "redis://server"

    def test_global_register_attribute_provider(self):
        """Test global register_attribute_provider function."""
        from ragguard.plugins.base import AttributeProvider
        from ragguard.plugins.registry import (
            get_attribute_provider,
            list_attribute_providers,
            register_attribute_provider,
        )

        class GlobalTestProvider(AttributeProvider):
            def __init__(self, url="default"):
                self.url = url
            def get_attributes(self, user_id): return {}

        register_attribute_provider("global_test_provider", GlobalTestProvider)

        assert "global_test_provider" in list_attribute_providers()
        provider = get_attribute_provider("global_test_provider", url="http://ldap")
        assert provider.url == "http://ldap"


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test all items in __all__ are accessible."""
        from ragguard.plugins import registry

        expected = [
            "PluginRegistry",
            "get_attribute_provider",
            "get_audit_sink",
            "get_cache_backend",
            "list_attribute_providers",
            "list_audit_sinks",
            "list_cache_backends",
            "register_attribute_provider",
            "register_audit_sink",
            "register_cache_backend",
        ]

        for name in expected:
            assert name in registry.__all__
            assert hasattr(registry, name)
