"""
Plugin Registry for RAGGuard

Central registration system for plugins. Enterprise packages and custom
plugins register their implementations here, and core code retrieves them
by name.
"""

from typing import Dict, Type

from .base import AttributeProvider, AuditSink, CacheBackend


class PluginRegistry:
    """
    Global registry for RAGGuard plugins.

    Plugins are registered by name and can be instantiated with configuration.
    This allows enterprise packages to extend RAGGuard without modifying core code.
    """

    def __init__(self):
        self._audit_sinks: Dict[str, Type[AuditSink]] = {}
        self._cache_backends: Dict[str, Type[CacheBackend]] = {}
        self._attribute_providers: Dict[str, Type[AttributeProvider]] = {}

    # Audit Sinks
    # -----------

    def register_audit_sink(self, name: str, sink_class: Type[AuditSink]) -> None:
        """
        Register an audit sink plugin.

        Args:
            name: Plugin name (e.g., "splunk", "datadog", "cloudwatch")
            sink_class: AuditSink implementation class

        Example:
            >>> registry.register_audit_sink("splunk", SplunkAuditSink)
        """
        if not issubclass(sink_class, AuditSink):
            raise TypeError(f"{sink_class} must be a subclass of AuditSink")
        self._audit_sinks[name] = sink_class

    def get_audit_sink(self, name: str, **config) -> AuditSink:
        """
        Create an audit sink instance.

        Args:
            name: Plugin name
            **config: Configuration passed to sink constructor

        Returns:
            Configured audit sink instance

        Raises:
            KeyError: If plugin name not found

        Example:
            >>> sink = registry.get_audit_sink(
            ...     "splunk",
            ...     hec_url="https://splunk.company.com:8088",
            ...     token="abc-123"
            ... )
        """
        if name not in self._audit_sinks:
            raise KeyError(
                f"Audit sink '{name}' not registered. "
                f"Available: {list(self._audit_sinks.keys())}"
            )
        return self._audit_sinks[name](**config)

    def list_audit_sinks(self) -> list[str]:
        """List all registered audit sink names."""
        return list(self._audit_sinks.keys())

    # Cache Backends
    # --------------

    def register_cache_backend(self, name: str, backend_class: Type[CacheBackend]) -> None:
        """
        Register a cache backend plugin.

        Args:
            name: Plugin name (e.g., "redis", "memcached", "dynamodb")
            backend_class: CacheBackend implementation class

        Example:
            >>> registry.register_cache_backend("redis", RedisCache)
        """
        if not issubclass(backend_class, CacheBackend):
            raise TypeError(f"{backend_class} must be a subclass of CacheBackend")
        self._cache_backends[name] = backend_class

    def get_cache_backend(self, name: str, **config) -> CacheBackend:
        """
        Create a cache backend instance.

        Args:
            name: Plugin name
            **config: Configuration passed to backend constructor

        Returns:
            Configured cache backend instance

        Raises:
            KeyError: If plugin name not found

        Example:
            >>> cache = registry.get_cache_backend(
            ...     "redis",
            ...     host="localhost",
            ...     port=6379,
            ...     db=0
            ... )
        """
        if name not in self._cache_backends:
            raise KeyError(
                f"Cache backend '{name}' not registered. "
                f"Available: {list(self._cache_backends.keys())}"
            )
        return self._cache_backends[name](**config)

    def list_cache_backends(self) -> list[str]:
        """List all registered cache backend names."""
        return list(self._cache_backends.keys())

    # Attribute Providers
    # -------------------

    def register_attribute_provider(
        self,
        name: str,
        provider_class: Type[AttributeProvider]
    ) -> None:
        """
        Register an attribute provider plugin.

        Args:
            name: Plugin name (e.g., "ldap", "okta", "auth0")
            provider_class: AttributeProvider implementation class

        Example:
            >>> registry.register_attribute_provider("ldap", LDAPAttributeProvider)
        """
        if not issubclass(provider_class, AttributeProvider):
            raise TypeError(f"{provider_class} must be a subclass of AttributeProvider")
        self._attribute_providers[name] = provider_class

    def get_attribute_provider(self, name: str, **config) -> AttributeProvider:
        """
        Create an attribute provider instance.

        Args:
            name: Plugin name
            **config: Configuration passed to provider constructor

        Returns:
            Configured attribute provider instance

        Raises:
            KeyError: If plugin name not found

        Example:
            >>> provider = registry.get_attribute_provider(
            ...     "ldap",
            ...     server="ldap.company.com",
            ...     base_dn="dc=company,dc=com"
            ... )
        """
        if name not in self._attribute_providers:
            raise KeyError(
                f"Attribute provider '{name}' not registered. "
                f"Available: {list(self._attribute_providers.keys())}"
            )
        return self._attribute_providers[name](**config)

    def list_attribute_providers(self) -> list[str]:
        """List all registered attribute provider names."""
        return list(self._attribute_providers.keys())


# Global registry instance
_registry = PluginRegistry()


# Convenience functions for global registry
# ------------------------------------------

def register_audit_sink(name: str, sink_class: Type[AuditSink]) -> None:
    """Register an audit sink plugin in the global registry."""
    _registry.register_audit_sink(name, sink_class)


def get_audit_sink(name: str, **config) -> AuditSink:
    """Get an audit sink from the global registry."""
    return _registry.get_audit_sink(name, **config)


def list_audit_sinks() -> list[str]:
    """List available audit sinks."""
    return _registry.list_audit_sinks()


def register_cache_backend(name: str, backend_class: Type[CacheBackend]) -> None:
    """Register a cache backend plugin in the global registry."""
    _registry.register_cache_backend(name, backend_class)


def get_cache_backend(name: str, **config) -> CacheBackend:
    """Get a cache backend from the global registry."""
    return _registry.get_cache_backend(name, **config)


def list_cache_backends() -> list[str]:
    """List available cache backends."""
    return _registry.list_cache_backends()


def register_attribute_provider(name: str, provider_class: Type[AttributeProvider]) -> None:
    """Register an attribute provider plugin in the global registry."""
    _registry.register_attribute_provider(name, provider_class)


def get_attribute_provider(name: str, **config) -> AttributeProvider:
    """Get an attribute provider from the global registry."""
    return _registry.get_attribute_provider(name, **config)


def list_attribute_providers() -> list[str]:
    """List available attribute providers."""
    return _registry.list_attribute_providers()


__all__ = [
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
