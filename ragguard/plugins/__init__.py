"""
Plugin System for RAGGuard

Provides extensibility interfaces for enterprise features:
- Audit sinks (Splunk, Datadog, CloudWatch, etc.)
- Cache backends (Redis, Memcached, etc.)
- Attribute providers (LDAP, external APIs, etc.)

These base interfaces are in the community edition.
Concrete implementations are available in ragguard-enterprise.
"""

from .base import AttributeProvider, AuditSink, CacheBackend
from .registry import (
    PluginRegistry,
    get_attribute_provider,
    get_audit_sink,
    get_cache_backend,
    list_attribute_providers,
    list_audit_sinks,
    list_cache_backends,
    register_attribute_provider,
    register_audit_sink,
    register_cache_backend,
)

__all__ = [
    # Base classes
    "AuditSink",
    "CacheBackend",
    "AttributeProvider",
    # Registry
    "PluginRegistry",
    "register_audit_sink",
    "get_audit_sink",
    "list_audit_sinks",
    "register_cache_backend",
    "get_cache_backend",
    "list_cache_backends",
    "register_attribute_provider",
    "get_attribute_provider",
    "list_attribute_providers",
]
