"""
Custom filter builders for complex, real-world scenarios.

When the built-in filter builder doesn't match your document schema or
permission model, use custom filter builders to inject your own logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from ..policy.models import Policy


class CustomFilterBuilder(ABC):
    """
    Base class for custom filter builders.

    Implement this when you need full control over how policies are
    translated to database filters.
    """

    @abstractmethod
    def build_filter(
        self,
        policy: Policy,
        user: dict[str, Any],
        backend: str
    ) -> Any:
        """
        Build a database-specific filter for the given user.

        Args:
            policy: The access control policy
            user: User context
            backend: Database backend ("qdrant", "pgvector", etc.)

        Returns:
            Database-specific filter object
        """
        pass


class FieldMappingFilterBuilder(CustomFilterBuilder):
    """
    Filter builder that maps policy fields to your document schema.

    Use this when your documents have different field names than
    what the policy uses.

    Example:
        # Policy uses: department, confidential
        # Your docs have: dept_code, is_sensitive

        mapping = {
            "department": "dept_code",
            "confidential": "is_sensitive"
        }

        builder = FieldMappingFilterBuilder(mapping)
    """

    def __init__(
        self,
        field_mapping: dict[str, str],
        value_transforms: Optional[dict[str, Callable[[Any], Any]]] = None
    ):
        """
        Initialize with field mapping.

        Args:
            field_mapping: Dict mapping policy field names to document field names
            value_transforms: Optional dict of field -> transform function
        """
        self.field_mapping = field_mapping
        self.value_transforms = value_transforms or {}

    def build_filter(
        self,
        policy: Policy,
        user: dict[str, Any],
        backend: str
    ) -> Any:
        """Build filter with field mapping applied."""
        # Import the standard builders
        if backend == "qdrant":
            from .builder import to_qdrant_filter
            # Modify policy with mapped fields
            mapped_policy = self._map_policy_fields(policy)
            return to_qdrant_filter(mapped_policy, user)
        elif backend == "pgvector":
            from .builder import to_pgvector_filter
            mapped_policy = self._map_policy_fields(policy)
            return to_pgvector_filter(mapped_policy, user)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _map_policy_fields(self, policy: Policy) -> Policy:
        """
        Create a new policy with mapped field names.

        Transforms document.X references in conditions to use the mapped field names.
        """

        if not self.field_mapping:
            return policy

        # Deep copy the policy dict to avoid mutating original
        policy_dict = policy.model_dump()

        # Transform conditions in each rule
        for rule in policy_dict.get("rules", []):
            # Map match conditions
            if rule.get("match"):
                rule["match"] = self._map_dict_fields(rule["match"])

            # Map allow conditions (string expressions)
            allow = rule.get("allow", {})
            if allow.get("conditions"):
                allow["conditions"] = [
                    self._map_condition_string(cond)
                    for cond in allow["conditions"]
                ]

        return Policy.model_validate(policy_dict)

    def _map_dict_fields(self, match_dict: dict[str, Any]) -> dict[str, Any]:
        """Map field names in a match dictionary."""
        result = {}
        for key, value in match_dict.items():
            mapped_key = self.field_mapping.get(key, key)
            result[mapped_key] = value
        return result

    def _map_condition_string(self, condition: str) -> str:
        """Map document.X field references in a condition string."""
        import re

        def replace_field(match: re.Match) -> str:
            field = match.group(1)
            mapped = self.field_mapping.get(field, field)
            return f"document.{mapped}"

        # Replace document.field_name with document.mapped_name
        return re.sub(r'document\.(\w+)', replace_field, condition)


class LambdaFilterBuilder(CustomFilterBuilder):
    """
    Filter builder that uses lambda functions for each backend.

    Use this for quick custom logic without creating a full class.

    ⚠️ SECURITY WARNING ⚠️
    When using LambdaFilterBuilder, YOU are responsible for:
    - Input validation on all user data
    - SQL injection prevention (use parameterized queries for pgvector)
    - Proper escaping of special characters
    - Validating field names and values

    RAGGuard does NOT automatically sanitize inputs in custom lambda functions.
    Improper usage can lead to security vulnerabilities.

    Example:
        def build_qdrant_filter(policy, user):
            # Your custom logic
            return Filter(...)

        def build_pgvector_filter(policy, user):
            # ✅ SAFE: Use parameterized queries
            return ("WHERE department = %s", [user.get("department")])

            # ❌ UNSAFE: Never do this (SQL injection risk)
            # return (f"WHERE department = '{user.get('department')}'", [])

        builder = LambdaFilterBuilder(
            qdrant=build_qdrant_filter,
            pgvector=build_pgvector_filter
        )

    Best Practices:
    - Always use parameterized queries for SQL backends
    - Validate all user inputs before using them
    - Use ACLFilterBuilder for common ACL patterns (safer)
    - Never directly interpolate user data into SQL strings
    """

    def __init__(
        self,
        qdrant: Optional[Callable[[Policy, dict[str, Any]], Any]] = None,
        pgvector: Optional[Callable[[Policy, dict[str, Any]], tuple[str, list]]] = None,
    ):
        """Initialize with lambda functions for each backend."""
        self.builders = {
            "qdrant": qdrant,
            "pgvector": pgvector,
        }

    def build_filter(
        self,
        policy: Policy,
        user: dict[str, Any],
        backend: str
    ) -> Any:
        """Build filter using registered lambda."""
        builder = self.builders.get(backend)
        if not builder:
            raise ValueError(f"No custom builder registered for backend: {backend}")

        return builder(policy, user)


class ACLFilterBuilder(CustomFilterBuilder):
    """
    Filter builder for ACL (Access Control List) based permissions.

    Use when documents have an explicit list of users/groups that can access them.

    Example document schema:
        {
            "text": "...",
            "acl": {
                "users": ["alice@company.com", "bob@company.com"],
                "groups": ["engineering", "product"],
                "public": false
            }
        }
    """

    def __init__(
        self,
        acl_field: str = "acl",
        users_field: str = "users",
        groups_field: str = "groups",
        public_field: str = "public",
        get_user_groups: Optional[Callable[[dict[str, Any]], list[str]]] = None
    ):
        """
        Initialize ACL filter builder.

        Args:
            acl_field: Field containing ACL object
            users_field: Field within ACL containing user list
            groups_field: Field within ACL containing group list
            public_field: Field within ACL indicating public access
            get_user_groups: Function to get groups for a user
        """
        # Validate field names to prevent SQL injection in pgvector queries
        self._validate_field_name(acl_field, "acl_field")
        self._validate_field_name(users_field, "users_field")
        self._validate_field_name(groups_field, "groups_field")
        self._validate_field_name(public_field, "public_field")

        self.acl_field = acl_field
        self.users_field = users_field
        self.groups_field = groups_field
        self.public_field = public_field
        self.get_user_groups = get_user_groups

    @staticmethod
    def _validate_field_name(value: str, param_name: str) -> None:
        """Validate field name to prevent SQL injection."""
        if not value or not isinstance(value, str):
            raise ValueError(f"{param_name} must be a non-empty string")
        if len(value) > 63:
            raise ValueError(f"{param_name} too long: {len(value)} chars (max 63)")
        # Allow alphanumeric, underscore, dollar sign (PostgreSQL standard)
        # Must start with letter or underscore (not digit - SQL standard)
        if not value[0].isalpha() and value[0] != '_':
            raise ValueError(
                f"Invalid {param_name}: must start with letter or underscore, got '{value[0]}'"
            )
        if not all(c.isalnum() or c in ('_', '$') for c in value):
            raise ValueError(
                f"Invalid {param_name}: contains invalid characters. "
                f"Only alphanumeric, underscore, and $ are allowed. Got: '{value}'"
            )

    @staticmethod
    def _validate_group_names(user_groups: list) -> None:
        """
        Validate group names to prevent injection attacks.

        Args:
            user_groups: List of group names to validate

        Raises:
            ValueError: If any group name is invalid
        """
        if not user_groups:
            return

        for group in user_groups:
            if not group or not isinstance(group, str):
                raise ValueError("Invalid group: must be non-empty string")
            if len(group) > 100:
                raise ValueError(f"Invalid group: too long ({len(group)} chars, max 100)")
            # Allow alphanumeric, dash, underscore, dot, colon, at, slash
            if not all(c.isalnum() or c in ('-', '_', '.', ':', '@', '/') for c in group):
                invalid_chars = [c for c in group if not (c.isalnum() or c in ('-', '_', '.', ':', '@', '/'))]
                raise ValueError(
                    f"Invalid group: contains invalid characters. "
                    f"Only alphanumeric and -_.:/@ are allowed. Invalid chars: {invalid_chars}"
                )

    def build_filter(
        self,
        policy: Policy,
        user: dict[str, Any],
        backend: str
    ) -> Any:
        """Build ACL-based filter."""
        user_id = user.get("id")
        user_groups = self.get_user_groups(user) if self.get_user_groups else []

        if backend == "qdrant":
            return self._build_qdrant_acl_filter(user_id, user_groups)
        elif backend == "pgvector":
            return self._build_pgvector_acl_filter(user_id, user_groups)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _build_qdrant_acl_filter(self, user_id: str, user_groups: list[str]) -> Any:
        """Build Qdrant filter for ACL."""
        # Validate list sizes to prevent DoS attacks
        MAX_GROUPS = 100
        if user_groups and len(user_groups) > MAX_GROUPS:
            raise ValueError(
                f"Too many user groups: {len(user_groups)} (max {MAX_GROUPS}). "
                f"This could cause performance issues."
            )

        # Validate group name content to prevent injection attacks
        self._validate_group_names(user_groups)

        try:
            from qdrant_client import models
        except ImportError:
            raise ImportError("qdrant-client required")

        conditions = []

        # Public documents
        public_path = f"{self.acl_field}.{self.public_field}"
        conditions.append(
            models.FieldCondition(
                key=public_path,
                match=models.MatchValue(value=True)
            )
        )

        # User explicitly in ACL
        if user_id:
            users_path = f"{self.acl_field}.{self.users_field}"
            conditions.append(
                models.FieldCondition(
                    key=users_path,
                    match=models.MatchAny(any=[user_id])
                )
            )

        # User's groups in ACL
        if user_groups:
            groups_path = f"{self.acl_field}.{self.groups_field}"
            conditions.append(
                models.FieldCondition(
                    key=groups_path,
                    match=models.MatchAny(any=user_groups)
                )
            )

        # Combine with OR
        return models.Filter(should=conditions) if conditions else None

    def _build_pgvector_acl_filter(
        self,
        user_id: str,
        user_groups: list[str]
    ) -> tuple[str, list[Any]]:
        """Build pgvector WHERE clause for ACL."""
        # Validate list sizes to prevent SQL DoS attacks
        MAX_GROUPS = 100
        if user_groups and len(user_groups) > MAX_GROUPS:
            raise ValueError(
                f"Too many user groups: {len(user_groups)} (max {MAX_GROUPS}). "
                f"This could cause SQL performance issues."
            )

        # Validate group name content to prevent injection attacks
        self._validate_group_names(user_groups)

        conditions: List[str] = []
        params: List[Any] = []

        # Public documents
        public_path = f"{self.acl_field}->>'{self.public_field}'"
        conditions.append(f"{public_path} = %s")
        params.append(True)

        # User in ACL (using JSONB contains operator)
        if user_id:
            users_path = f"{self.acl_field}->'{self.users_field}'"
            conditions.append(f"{users_path} @> %s::jsonb")
            params.append(f'["{user_id}"]')

        # Groups in ACL
        if user_groups:
            groups_path = f"{self.acl_field}->'{self.groups_field}'"
            group_conditions = " OR ".join([f"{groups_path} @> %s::jsonb"] * len(user_groups))
            conditions.append(f"({group_conditions})")
            params.extend([f'["{group}"]' for group in user_groups])

        # Combine with OR
        combined = " OR ".join(f"({cond})" for cond in conditions)
        return (f"WHERE {combined}", params)


class HybridFilterBuilder(CustomFilterBuilder):
    """
    Combines the standard filter builder with custom extensions.

    Use when you want the standard policy-based filtering plus
    additional custom logic.
    """

    def __init__(
        self,
        additional_filters: dict[str, Callable[[dict[str, Any]], Any]]
    ):
        """
        Initialize hybrid builder.

        Args:
            additional_filters: Dict of backend -> filter function
                Function receives user context and returns additional filter
        """
        self.additional_filters = additional_filters

    def build_filter(
        self,
        policy: Policy,
        user: dict[str, Any],
        backend: str
    ) -> Any:
        """Build combined filter."""
        # Get standard filter
        if backend == "qdrant":
            from .builder import to_qdrant_filter
            standard_filter = to_qdrant_filter(policy, user)
        elif backend == "pgvector":
            from .builder import to_pgvector_filter
            standard_filter = to_pgvector_filter(policy, user)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Get additional filter
        additional_func = self.additional_filters.get(backend)
        if not additional_func:
            return standard_filter

        additional_filter = additional_func(user)

        # Combine filters (AND logic)
        return self._combine_filters(standard_filter, additional_filter, backend)

    def _combine_filters(self, filter1: Any, filter2: Any, backend: str) -> Any:
        """Combine two filters with AND logic."""
        if backend == "qdrant":
            from qdrant_client import models
            # Combine with AND
            if filter1 is None:
                return filter2
            if filter2 is None:
                return filter1
            return models.Filter(must=[filter1, filter2])

        elif backend == "pgvector":
            # Both should be (where_clause, params) tuples
            clause1, params1 = filter1
            clause2, params2 = filter2

            if not clause1:
                return (clause2, params2)
            if not clause2:
                return (clause1, params1)

            # Combine: WHERE (clause1) AND (clause2)
            combined_clause = f"{clause1} AND ({clause2.replace('WHERE ', '')})"
            combined_params = params1 + params2
            return (combined_clause, combined_params)

        return filter1
