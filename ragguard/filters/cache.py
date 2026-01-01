"""
Filter caching infrastructure for RAGGuard performance optimization.

This module provides thread-safe caching of database filter objects to avoid
rebuilding them on every query. The cache uses an LRU eviction strategy with
configurable size limits.
"""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from ..types import CachedFilter, HasModelDump

if TYPE_CHECKING:
    from ..policy.models import Policy


class FilterCache:
    """
    Thread-safe LRU cache for database filter objects.

    The cache stores pre-built filters keyed by user context and policy,
    enabling O(1) retrieval for repeated queries from the same user.

    Features:
    - LRU eviction when cache reaches max size
    - Thread-safe operations using RLock
    - Cache hit/miss statistics tracking
    - Explicit invalidation API

    Example:
        >>> cache = FilterCache(max_size=1000)
        >>> cache.set("user123:policy_abc:qdrant", filter_object)
        >>> cached_filter = cache.get("user123:policy_abc:qdrant")
        >>> cache.get_stats()
        {'hits': 145, 'misses': 12, 'hit_rate': 0.923, 'size': 157}
    """

    def __init__(self, max_size: int = 1000, lock_timeout: float = 5.0):
        """
        Initialize the filter cache.

        Args:
            max_size: Maximum number of entries in the cache before eviction.
                     Default is 1000, which should handle ~100KB of cached filters.
            lock_timeout: Maximum time (in seconds) to wait for lock acquisition.
                         Default is 5.0 seconds. Use -1 for no timeout (blocking).
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self._max_size = max_size
        self._cache: OrderedDict[str, CachedFilter] = OrderedDict()
        self._lock = threading.RLock()
        self._lock_timeout = lock_timeout if lock_timeout >= 0 else None

        # Statistics
        self._hits = 0
        self._misses = 0

    def _acquire_lock(self) -> bool:
        """
        Acquire the cache lock with timeout.

        Returns:
            True if lock acquired, False if timeout

        Raises:
            RuntimeError: If lock acquisition times out
        """
        acquired = self._lock.acquire(timeout=self._lock_timeout)
        if not acquired:
            raise RuntimeError(
                f"Failed to acquire filter cache lock within {self._lock_timeout}s timeout. "
                f"This may indicate high lock contention or a deadlock."
            )
        return acquired

    def get(self, key: str) -> Optional[CachedFilter]:
        """
        Retrieve a cached filter by key.

        This operation is thread-safe and updates the LRU order.

        Args:
            key: Cache key (typically "backend:policy_hash:user_hash")

        Returns:
            The cached filter object, or None if not found

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            if key in self._cache:
                # Move to end (mark as recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None
        finally:
            self._lock.release()

    def set(self, key: str, value: CachedFilter) -> None:
        """
        Store a filter in the cache.

        If the cache is full, the least recently used entry is evicted.
        This operation is thread-safe.

        Args:
            key: Cache key
            value: Filter object to cache

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new entry
                if len(self._cache) >= self._max_size:
                    # Evict least recently used (first item)
                    self._cache.popitem(last=False)

                self._cache[key] = value
        finally:
            self._lock.release()

    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from the cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if the key was found and removed, False otherwise

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
        finally:
            self._lock.release()

    def invalidate_all(self) -> None:
        """
        Clear all entries from the cache.

        This is useful when the policy is updated and all cached filters
        should be regenerated.

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            self._cache.clear()
            # Keep statistics - they're useful for understanding cache effectiveness
        finally:
            self._lock.release()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Percentage of requests that hit the cache
            - size: Current number of entries in cache
            - max_size: Maximum cache size

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self._max_size
            }
        finally:
            self._lock.release()

    def reset_stats(self) -> None:
        """
        Reset cache statistics counters.

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            self._hits = 0
            self._misses = 0
        finally:
            self._lock.release()

    def __len__(self) -> int:
        """
        Return the current number of cached entries.

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            return len(self._cache)
        finally:
            self._lock.release()

    def __contains__(self, key: str) -> bool:
        """
        Check if a key is in the cache.

        Raises:
            RuntimeError: If lock acquisition times out
        """
        self._acquire_lock()
        try:
            return key in self._cache
        finally:
            self._lock.release()


class CacheKeyBuilder:
    """
    Helper class for building cache keys from policy, user context, and backend.

    Cache keys must be:
    1. Stable - same inputs always produce same key
    2. Efficient - quick to compute
    3. Selective - only include fields that affect the filter
    """

    @staticmethod
    def build_key(
        backend: str,
        policy_hash: str,
        user: Dict[str, Any],
        relevant_user_fields: Set[str]
    ) -> str:
        """
        Build a cache key from the components.

        Args:
            backend: Database backend (e.g., "qdrant", "pgvector")
            policy_hash: Hash of the policy object
            user: User context dictionary
            relevant_user_fields: Set of user fields that affect filtering
                                 (e.g., {"roles", "department", "team"})

        Returns:
            Cache key string
        """
        # Build user hash from only relevant fields
        user_hash = CacheKeyBuilder._compute_user_hash(user, relevant_user_fields)

        # Combine into cache key
        return f"{backend}:{policy_hash}:{user_hash}"

    @staticmethod
    def _compute_user_hash(
        user: Dict[str, Any],
        relevant_fields: Set[str]
    ) -> str:
        """
        Compute a stable hash of user context.

        Only includes fields that are referenced in the policy conditions,
        to maximize cache hit rate. For example, if the policy only checks
        user.roles and user.department, we ignore user.id, user.email, etc.

        Security: Uses SHA-256 instead of MD5 to prevent collision attacks
        that could allow unauthorized access to cached filters.

        Args:
            user: User context dictionary
            relevant_fields: Set of field names to include in hash

        Returns:
            Hex string hash of relevant user fields
        """
        # Extract only relevant fields
        relevant_data = {}
        for field in sorted(relevant_fields):  # Sort for stability
            value = CacheKeyBuilder._get_nested_field(user, field)
            if value is not None:
                # Normalize lists and sets to sorted tuples for consistent hashing
                if isinstance(value, (list, set)):
                    # Preserve type information to prevent cache poisoning
                    # Convert to list of (type, value) tuples to avoid collisions
                    # e.g., ['admin', 123] != ['admin', '123']
                    typed_values = [(type(v).__name__, str(v)) for v in value]
                    value = tuple(sorted(typed_values))
                relevant_data[field] = value

        # Convert to stable JSON string
        json_str = json.dumps(relevant_data, sort_keys=True, default=str)

        # Hash to fixed-length string using SHA-256 (secure against collision attacks)
        # Use first 32 chars for balance between uniqueness and key length
        return hashlib.sha256(json_str.encode()).hexdigest()[:32]

    @staticmethod
    def _get_nested_field(obj: Dict[str, Any], field_path: str) -> Any:
        """
        Get a nested field from a dictionary using dot notation.

        Args:
            obj: Dictionary to extract from
            field_path: Field path like "metadata.team.id"

        Returns:
            The field value, or None if not found
        """
        value = obj
        for key in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        return value


def compute_policy_hash(policy: HasModelDump) -> str:
    """
    Compute a stable hash of a policy object.

    This hash is used in cache keys to ensure that when the policy changes,
    cached filters are invalidated.

    Args:
        policy: Policy object (must have model_dump method)

    Returns:
        Hex string hash (first 16 characters of SHA-256)
    """
    # Use policy's JSON representation for stable hashing
    # Pydantic's model_dump_json() doesn't have sort_keys, so we use model_dump() + json.dumps()
    policy_dict = policy.model_dump()
    policy_json = json.dumps(policy_dict, sort_keys=True)
    hash_full = hashlib.sha256(policy_json.encode()).hexdigest()
    return hash_full[:16]  # Use first 16 chars for readability


def extract_user_fields_from_policy(
    policy: "Policy",
    compiled_conditions: Optional[Dict[int, list]] = None
) -> Set[str]:
    """
    Extract the set of user fields referenced in a policy.

    This is used to determine which user context fields affect filtering,
    allowing us to build more efficient cache keys.

    SECURITY: This function must capture ALL user fields that could affect
    access decisions. Missing a field could lead to cache poisoning where
    users with different permissions share cached filters.

    Args:
        policy: Policy object with rules
        compiled_conditions: Optional pre-compiled conditions from PolicyEngine
                            (more accurate than regex parsing)

    Returns:
        Set of field names like {"roles", "department", "metadata.team"}
    """
    import re

    fields = set()

    # Always include roles since they're checked in allow.roles
    fields.add("roles")

    # Also include 'id' as it's commonly used for identity checks
    fields.add("id")

    # Extract from compiled conditions if available (more accurate)
    if compiled_conditions:
        fields.update(_extract_fields_from_compiled(compiled_conditions))

    # Also extract from condition strings as fallback
    # This catches any fields that might not have been compiled
    for rule in policy.rules:
        if rule.allow.conditions:
            for condition in rule.allow.conditions:
                # Find all "user.X" references in condition strings
                # Matches: user.field, user.nested.field, etc.
                matches = re.findall(r'user\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', condition)
                fields.update(matches)

    return fields


def _extract_fields_from_compiled(
    compiled_conditions: Dict[int, list]
) -> Set[str]:
    """
    Extract user fields from compiled condition AST.

    This is more accurate than regex parsing as it uses the actual
    parsed condition structure.

    Args:
        compiled_conditions: Dict of rule_index -> list of CompiledCondition/CompiledExpression

    Returns:
        Set of user field names
    """
    from ..policy.compiler import CompiledCondition, CompiledExpression, ValueType

    fields = set()

    def extract_from_node(node) -> None:
        """Recursively extract user fields from a compiled node."""
        if isinstance(node, CompiledCondition):
            # Check left operand
            if node.left.value_type == ValueType.USER_FIELD:
                field_path = ".".join(node.left.field_path)
                fields.add(field_path)
            # Check right operand
            if node.right and node.right.value_type == ValueType.USER_FIELD:
                field_path = ".".join(node.right.field_path)
                fields.add(field_path)
        elif isinstance(node, CompiledExpression):
            # Recurse into child nodes
            for child in node.children:
                extract_from_node(child)

    for condition_list in compiled_conditions.values():
        for node in condition_list:
            extract_from_node(node)

    return fields


def validate_cache_key_completeness(
    user: Dict[str, Any],
    relevant_fields: Set[str],
    policy_conditions: Optional[list] = None
) -> bool:
    """
    Validate that all user fields affecting access are captured in cache key.

    SECURITY: This validation helps detect potential cache poisoning scenarios
    where the cache key might not include all relevant user attributes.

    Args:
        user: User context dictionary
        relevant_fields: Fields that will be included in cache key
        policy_conditions: Optional list of condition strings to validate against

    Returns:
        True if validation passes, False if there might be uncaptured fields

    Note:
        This is a best-effort validation. For maximum security, consider
        using conservative_cache_key_mode which includes all user fields.
    """
    if not policy_conditions:
        return True

    import re

    # Check all conditions for user field references
    for condition in policy_conditions:
        matches = re.findall(r'user\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)', condition)
        for field in matches:
            if field not in relevant_fields:
                return False

    return True
