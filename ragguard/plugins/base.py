"""
Base interfaces for RAGGuard plugins.

These abstract classes define the contracts for extending RAGGuard with
custom integrations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AuditSink(ABC):
    """
    Plugin interface for audit log destinations.

    Audit sinks receive structured log entries and send them to external
    systems for compliance, monitoring, and analysis.

    Examples:
        - SplunkAuditSink
        - DatadogAuditSink
        - CloudWatchAuditSink
        - ElasticsearchAuditSink
    """

    @abstractmethod
    def write(self, entry: Dict[str, Any]) -> None:
        """
        Write an audit log entry.

        Args:
            entry: Structured log entry containing:
                - timestamp: ISO 8601 timestamp
                - user_id: User identifier
                - query: Query string or "[vector]"
                - results_returned: Number of results
                - filter: Applied filter expression
                - Additional custom fields

        Raises:
            Exception: If write fails (should be caught by caller)
        """
        pass

    def close(self) -> None:
        """
        Close any open connections or resources.

        Called when the audit logger is shutting down.
        Override if your sink needs cleanup.
        """
        pass


class CacheBackend(ABC):
    """
    Plugin interface for distributed caching.

    Cache backends store filter objects across multiple instances for
    better performance in distributed deployments.

    Examples:
        - RedisCache
        - MemcachedCache
        - DynamoDBCache
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (will be serialized)
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all entries from the cache.

        Use with caution - this affects all cached data.
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with backend-specific statistics.
            Suggested fields: hits, misses, size, evictions
        """
        return {}

    def close(self) -> None:
        """
        Close connections and cleanup resources.

        Override if your backend needs cleanup.
        """
        pass


class AttributeProvider(ABC):
    """
    Plugin interface for user attribute enrichment.

    Attribute providers fetch additional user attributes from external
    systems to enrich the user context for policy evaluation.

    Examples:
        - LDAPAttributeProvider
        - OktaAttributeProvider
        - Auth0AttributeProvider
        - DatabaseAttributeProvider

    Use cases:
        - Fetch user groups from LDAP/Active Directory
        - Get user department from HR system
        - Retrieve clearance level from security database
    """

    @abstractmethod
    def get_attributes(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch additional attributes for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of attribute name -> value
            Example: {"department": "engineering", "groups": ["dev", "ops"]}

        Note:
            - Should cache results internally to avoid repeated API calls
            - Should return empty dict (not None) if user not found
            - Should raise exception if external system is unavailable
        """
        pass

    def enrich_user(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a user context with additional attributes.

        Args:
            user: Existing user context (must have 'id' field)

        Returns:
            Enriched user context (merged attributes)

        Example:
            >>> user = {"id": "alice"}
            >>> enriched = provider.enrich_user(user)
            >>> # enriched = {"id": "alice", "department": "eng", "groups": [...]}
        """
        user_id = user.get("id")
        if not user_id:
            return user

        attributes = self.get_attributes(user_id)
        return {**user, **attributes}

    def close(self) -> None:
        """
        Close connections and cleanup resources.

        Override if your provider needs cleanup.
        """
        pass


# Type aliases for plugin factories
AuditSinkFactory = type[AuditSink]
CacheBackendFactory = type[CacheBackend]
AttributeProviderFactory = type[AttributeProvider]


__all__ = [
    "AttributeProvider",
    "AttributeProviderFactory",
    "AuditSink",
    "AuditSinkFactory",
    "CacheBackend",
    "CacheBackendFactory",
]
