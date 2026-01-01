"""
Permission resolvers for complex, real-world authorization scenarios.

Resolvers allow you to integrate with external permission systems,
implement custom business logic, and handle scenarios beyond simple
field matching.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class PermissionResolver(ABC):
    """
    Base class for permission resolvers.

    A resolver is called during filter building to determine if a user
    should have access. This allows integration with:
    - LDAP/Active Directory
    - Graph APIs (Microsoft Graph, Google Workspace)
    - Custom authorization services
    - Complex business logic
    """

    @abstractmethod
    def can_access(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> bool:
        """
        Determine if user can access resources matching this rule.

        Args:
            user: User context
            rule_name: Name of the rule being evaluated
            rule_context: Additional context (match conditions, etc.)

        Returns:
            True if user should have access
        """
        pass

    @abstractmethod
    def get_filter_params(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Get filter parameters to apply for this user/rule.

        Returns None if user should not have access.
        Returns dict of field->value mappings to filter on.

        Example:
            # User belongs to organizations ["org1", "org2"]
            return {"organization_id": ["org1", "org2"]}

            # This becomes: WHERE organization_id IN ('org1', 'org2')
        """
        pass


class RoleHierarchyResolver(PermissionResolver):
    """
    Resolver that understands role hierarchies.

    Example:
        hierarchy = {
            "admin": ["manager", "employee", "intern"],
            "manager": ["employee", "intern"],
            "employee": ["intern"]
        }

        # User with "manager" role also has "employee" and "intern" permissions
    """

    def __init__(self, hierarchy: dict[str, list[str]]):
        """
        Initialize with role hierarchy.

        Args:
            hierarchy: Dict mapping role -> list of inherited roles
        """
        self.hierarchy = hierarchy

    def get_effective_roles(self, user: dict[str, Any]) -> set[str]:
        """Get all roles a user has (direct + inherited)."""
        user_roles = user.get("roles", [])
        if isinstance(user_roles, str):
            user_roles = [user_roles]

        effective_roles = set(user_roles)

        # Add inherited roles
        for role in list(effective_roles):
            if role in self.hierarchy:
                effective_roles.update(self.hierarchy[role])

        return effective_roles

    def can_access(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> bool:
        """Check if user has required roles (including inherited)."""
        required_roles = rule_context.get("required_roles", [])
        if not required_roles:
            return True

        user_roles = self.get_effective_roles(user)
        return any(role in user_roles for role in required_roles)

    def get_filter_params(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Return None - role checks don't add filter params."""
        return None if self.can_access(user, rule_name, rule_context) else {}


class OrganizationResolver(PermissionResolver):
    """
    Resolver for multi-tenant organizations.

    Users can belong to multiple organizations and see documents
    from any of their organizations.
    """

    def __init__(
        self,
        get_user_organizations: Callable[[str], list[str]],
        org_field: str = "organization_id"
    ):
        """
        Initialize organization resolver.

        Args:
            get_user_organizations: Function that takes user_id and returns list of org IDs
            org_field: Document field containing organization ID
        """
        self.get_user_organizations = get_user_organizations
        self.org_field = org_field

    def can_access(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> bool:
        """User can access if they belong to at least one org."""
        user_id = user.get("id")
        if not user_id:
            return False

        orgs = self.get_user_organizations(user_id)
        return len(orgs) > 0

    def get_filter_params(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Return organization filter."""
        user_id = user.get("id")
        if not user_id:
            return {}

        orgs = self.get_user_organizations(user_id)
        if not orgs:
            return {}

        return {self.org_field: orgs}


# GraphPermissionResolver moved to ragguard-enterprise
# It requires external API integrations (Microsoft Graph, Google Workspace)
# which are enterprise features.
# Import from: ragguard_enterprise.policy.resolvers


class TimeBasedResolver(PermissionResolver):
    """
    Resolver for time-based permissions.

    Documents can have expiration times or access windows.
    """

    def __init__(self, current_time_provider: Optional[Callable[[], Any]] = None):
        """
        Initialize time-based resolver.

        Args:
            current_time_provider: Function that returns current time (for testing)
        """
        from datetime import datetime, timezone
        self.get_current_time = current_time_provider or (lambda: datetime.now(timezone.utc))

    def can_access(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> bool:
        """Always allow - time filtering happens in database."""
        return True

    def get_filter_params(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Add time-based filter."""
        current_time = self.get_current_time()

        # Return filter that checks:
        # - expires_at is None OR expires_at > now
        # - available_from is None OR available_from <= now
        return {
            "_time_filter": {
                "current_time": current_time,
                "check_expiry": True
            }
        }


class CompositeResolver(PermissionResolver):
    """
    Combines multiple resolvers with AND logic.

    All resolvers must approve for access to be granted.
    """

    def __init__(self, resolvers: list[PermissionResolver]):
        """Initialize with list of resolvers."""
        self.resolvers = resolvers

    def can_access(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> bool:
        """All resolvers must approve."""
        return all(
            resolver.can_access(user, rule_name, rule_context)
            for resolver in self.resolvers
        )

    def get_filter_params(
        self,
        user: dict[str, Any],
        rule_name: str,
        rule_context: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Merge filter params from all resolvers."""
        if not self.can_access(user, rule_name, rule_context):
            return {}

        merged = {}
        for resolver in self.resolvers:
            params = resolver.get_filter_params(user, rule_name, rule_context)
            if params is None:
                continue
            if params == {}:  # Explicit deny
                return {}
            merged.update(params)

        return merged if merged else None


# Type alias for custom resolver functions
PermissionResolverFunc = Callable[[dict[str, Any], str, dict[str, Any]], bool]
