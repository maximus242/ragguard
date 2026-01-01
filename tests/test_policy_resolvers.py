"""
Tests for permission resolvers.

These tests cover the resolver extensibility layer that allows integration
with external permission systems and custom business logic.
"""

from datetime import datetime, timedelta

import pytest

# Test RoleHierarchyResolver
# ===========================

def test_role_hierarchy_resolver_direct_role():
    """Test RoleHierarchyResolver with direct role match."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {
        "admin": ["manager", "employee", "intern"],
        "manager": ["employee", "intern"],
        "employee": ["intern"]
    }

    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": ["employee"]}
    rule_context = {"required_roles": ["employee"]}

    # User has direct role
    assert resolver.can_access(user, "test_rule", rule_context)


def test_role_hierarchy_resolver_inherited_role():
    """Test RoleHierarchyResolver with inherited role."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {
        "admin": ["manager", "employee", "intern"],
        "manager": ["employee", "intern"],
        "employee": ["intern"]
    }

    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": ["manager"]}
    rule_context = {"required_roles": ["employee"]}

    # Manager inherits employee role
    assert resolver.can_access(user, "test_rule", rule_context)


def test_role_hierarchy_resolver_admin_has_all():
    """Test that admin inherits all roles."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {
        "admin": ["manager", "employee", "intern"],
        "manager": ["employee", "intern"],
        "employee": ["intern"]
    }

    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": ["admin"]}

    # Admin should have all roles
    effective_roles = resolver.get_effective_roles(user)
    assert "admin" in effective_roles
    assert "manager" in effective_roles
    assert "employee" in effective_roles
    assert "intern" in effective_roles


def test_role_hierarchy_resolver_no_match():
    """Test RoleHierarchyResolver denies when no role match."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {
        "admin": ["manager", "employee"],
        "manager": ["employee"]
    }

    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": ["intern"]}
    rule_context = {"required_roles": ["manager"]}

    # Intern does not have manager role
    assert not resolver.can_access(user, "test_rule", rule_context)


def test_role_hierarchy_resolver_single_string_role():
    """Test RoleHierarchyResolver handles single role as string."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {"admin": ["manager"]}

    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": "admin"}  # String, not list

    effective_roles = resolver.get_effective_roles(user)
    assert "admin" in effective_roles
    assert "manager" in effective_roles


def test_role_hierarchy_resolver_get_filter_params():
    """Test RoleHierarchyResolver get_filter_params."""
    from ragguard.policy.resolvers import RoleHierarchyResolver

    hierarchy = {"admin": ["manager"]}
    resolver = RoleHierarchyResolver(hierarchy)

    user = {"id": "alice", "roles": ["admin"]}
    rule_context = {"required_roles": ["manager"]}

    # Should return None since role checks don't add filter params
    assert resolver.get_filter_params(user, "test_rule", rule_context) is None


# Test OrganizationResolver
# ==========================

def test_organization_resolver_single_org():
    """Test OrganizationResolver with single organization."""
    from ragguard.policy.resolvers import OrganizationResolver

    def get_user_orgs(user_id):
        return ["org1"]

    resolver = OrganizationResolver(get_user_orgs)

    user = {"id": "alice"}
    rule_context = {}

    # User can access
    assert resolver.can_access(user, "test_rule", rule_context)

    # Filter params should include organization
    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {"organization_id": ["org1"]}


def test_organization_resolver_multiple_orgs():
    """Test OrganizationResolver with multiple organizations."""
    from ragguard.policy.resolvers import OrganizationResolver

    def get_user_orgs(user_id):
        return ["org1", "org2", "org3"]

    resolver = OrganizationResolver(get_user_orgs)

    user = {"id": "bob"}
    rule_context = {}

    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {"organization_id": ["org1", "org2", "org3"]}


def test_organization_resolver_no_orgs():
    """Test OrganizationResolver denies when user has no orgs."""
    from ragguard.policy.resolvers import OrganizationResolver

    def get_user_orgs(user_id):
        return []

    resolver = OrganizationResolver(get_user_orgs)

    user = {"id": "alice"}
    rule_context = {}

    # No orgs = no access
    assert not resolver.can_access(user, "test_rule", rule_context)

    # Filter params should be empty dict (explicit deny)
    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {}


def test_organization_resolver_no_user_id():
    """Test OrganizationResolver denies when no user ID."""
    from ragguard.policy.resolvers import OrganizationResolver

    def get_user_orgs(user_id):
        return ["org1"]

    resolver = OrganizationResolver(get_user_orgs)

    user = {}  # No ID
    rule_context = {}

    assert not resolver.can_access(user, "test_rule", rule_context)
    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {}


def test_organization_resolver_custom_field():
    """Test OrganizationResolver with custom org field name."""
    from ragguard.policy.resolvers import OrganizationResolver

    def get_user_orgs(user_id):
        return ["tenant1", "tenant2"]

    resolver = OrganizationResolver(get_user_orgs, org_field="tenant_id")

    user = {"id": "alice"}
    rule_context = {}

    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {"tenant_id": ["tenant1", "tenant2"]}


# Test GraphPermissionResolver (Enterprise)
# ==========================================
# GraphPermissionResolver moved to ragguard-enterprise

def test_graph_permission_resolver_check_permission():
    """Test GraphPermissionResolver with permission check."""
    try:
        from ragguard_enterprise.policy.resolvers import GraphPermissionResolver
    except ImportError:
        pytest.skip("GraphPermissionResolver requires ragguard-enterprise")

    def check_perm(user_id, resource_id, permission):
        # Alice has read permission on resource1
        return user_id == "alice" and resource_id == "resource1" and permission == "read"

    resolver = GraphPermissionResolver(check_permission=check_perm)

    user = {"id": "alice"}
    rule_context = {"resource_id": "resource1", "permission": "read"}

    assert resolver.can_access(user, "test_rule", rule_context)


def test_graph_permission_resolver_deny():
    """Test GraphPermissionResolver denies when no permission."""
    try:
        from ragguard_enterprise.policy.resolvers import GraphPermissionResolver
    except ImportError:
        pytest.skip("GraphPermissionResolver requires ragguard-enterprise")

    def check_perm(user_id, resource_id, permission):
        return False

    resolver = GraphPermissionResolver(check_permission=check_perm)

    user = {"id": "alice"}
    rule_context = {"resource_id": "resource1"}

    assert not resolver.can_access(user, "test_rule", rule_context)


def test_graph_permission_resolver_with_groups():
    """Test GraphPermissionResolver with user groups."""
    try:
        from ragguard_enterprise.policy.resolvers import GraphPermissionResolver
    except ImportError:
        pytest.skip("GraphPermissionResolver requires ragguard-enterprise")

    def check_perm(user_id, resource_id, permission):
        return True

    def get_groups(user_id):
        return ["group1", "group2"]

    resolver = GraphPermissionResolver(
        check_permission=check_perm,
        get_user_groups=get_groups
    )

    user = {"id": "alice"}
    rule_context = {"resource_id": "resource1"}

    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params == {"groups": ["group1", "group2"]}


def test_graph_permission_resolver_no_groups_func():
    """Test GraphPermissionResolver without groups function."""
    try:
        from ragguard_enterprise.policy.resolvers import GraphPermissionResolver
    except ImportError:
        pytest.skip("GraphPermissionResolver requires ragguard-enterprise")

    def check_perm(user_id, resource_id, permission):
        return True

    resolver = GraphPermissionResolver(check_permission=check_perm)

    user = {"id": "alice"}
    rule_context = {"resource_id": "resource1"}

    # Should return None when no groups function
    params = resolver.get_filter_params(user, "test_rule", rule_context)
    assert params is None


def test_graph_permission_resolver_missing_resource():
    """Test GraphPermissionResolver denies when resource missing."""
    try:
        from ragguard_enterprise.policy.resolvers import GraphPermissionResolver
    except ImportError:
        pytest.skip("GraphPermissionResolver requires ragguard-enterprise")

    def check_perm(user_id, resource_id, permission):
        return True

    resolver = GraphPermissionResolver(check_permission=check_perm)

    user = {"id": "alice"}
    rule_context = {}  # No resource_id

    assert not resolver.can_access(user, "test_rule", rule_context)


# Test TimeBasedResolver
# =======================

def test_time_based_resolver_always_allows():
    """Test TimeBasedResolver always allows access."""
    from ragguard.policy.resolvers import TimeBasedResolver

    resolver = TimeBasedResolver()

    user = {"id": "alice"}
    rule_context = {}

    # Always allows - time filtering happens in database
    assert resolver.can_access(user, "test_rule", rule_context)


def test_time_based_resolver_adds_time_filter():
    """Test TimeBasedResolver adds time filter parameters."""
    from ragguard.policy.resolvers import TimeBasedResolver

    current_time = datetime(2025, 1, 15, 12, 0, 0)
    resolver = TimeBasedResolver(current_time_provider=lambda: current_time)

    user = {"id": "alice"}
    rule_context = {}

    params = resolver.get_filter_params(user, "test_rule", rule_context)

    assert "_time_filter" in params
    assert params["_time_filter"]["current_time"] == current_time
    assert params["_time_filter"]["check_expiry"] is True


def test_time_based_resolver_default_time():
    """Test TimeBasedResolver uses current time by default."""
    from ragguard.policy.resolvers import TimeBasedResolver

    resolver = TimeBasedResolver()

    user = {"id": "alice"}
    rule_context = {}

    params = resolver.get_filter_params(user, "test_rule", rule_context)

    # Should have a time filter with current time
    assert "_time_filter" in params
    assert "current_time" in params["_time_filter"]


# Test CompositeResolver
# =======================

def test_composite_resolver_all_approve():
    """Test CompositeResolver when all resolvers approve."""
    from ragguard.policy.resolvers import (
        CompositeResolver,
        OrganizationResolver,
        RoleHierarchyResolver,
    )

    hierarchy = {"admin": ["manager"]}
    role_resolver = RoleHierarchyResolver(hierarchy)

    def get_orgs(user_id):
        return ["org1"]
    org_resolver = OrganizationResolver(get_orgs)

    composite = CompositeResolver([role_resolver, org_resolver])

    user = {"id": "alice", "roles": ["admin"]}
    rule_context = {"required_roles": ["manager"]}

    # Both approve
    assert composite.can_access(user, "test_rule", rule_context)


def test_composite_resolver_one_denies():
    """Test CompositeResolver denies when one resolver denies."""
    from ragguard.policy.resolvers import (
        CompositeResolver,
        OrganizationResolver,
        RoleHierarchyResolver,
    )

    hierarchy = {"admin": ["manager"]}
    role_resolver = RoleHierarchyResolver(hierarchy)

    def get_orgs(user_id):
        return []  # No orgs = deny
    org_resolver = OrganizationResolver(get_orgs)

    composite = CompositeResolver([role_resolver, org_resolver])

    user = {"id": "alice", "roles": ["admin"]}
    rule_context = {"required_roles": ["manager"]}

    # Org resolver denies
    assert not composite.can_access(user, "test_rule", rule_context)


def test_composite_resolver_merge_params():
    """Test CompositeResolver merges filter params from all resolvers."""
    from ragguard.policy.resolvers import CompositeResolver, OrganizationResolver, TimeBasedResolver

    def get_orgs(user_id):
        return ["org1", "org2"]
    org_resolver = OrganizationResolver(get_orgs)

    current_time = datetime(2025, 1, 15, 12, 0, 0)
    time_resolver = TimeBasedResolver(current_time_provider=lambda: current_time)

    composite = CompositeResolver([org_resolver, time_resolver])

    user = {"id": "alice"}
    rule_context = {}

    params = composite.get_filter_params(user, "test_rule", rule_context)

    # Should have both org and time filters
    assert "organization_id" in params
    assert params["organization_id"] == ["org1", "org2"]
    assert "_time_filter" in params
    assert params["_time_filter"]["current_time"] == current_time


def test_composite_resolver_explicit_deny():
    """Test CompositeResolver handles explicit deny (empty dict)."""
    from ragguard.policy.resolvers import CompositeResolver, OrganizationResolver

    def get_orgs(user_id):
        return []  # Returns empty list = explicit deny
    org_resolver = OrganizationResolver(get_orgs)

    composite = CompositeResolver([org_resolver])

    user = {"id": "alice"}
    rule_context = {}

    params = composite.get_filter_params(user, "test_rule", rule_context)

    # Empty dict means explicit deny
    assert params == {}


def test_composite_resolver_empty_list():
    """Test CompositeResolver with empty list of resolvers."""
    from ragguard.policy.resolvers import CompositeResolver

    composite = CompositeResolver([])

    user = {"id": "alice"}
    rule_context = {}

    # All (zero) resolvers approve
    assert composite.can_access(user, "test_rule", rule_context)

    # Should return None (no filters)
    params = composite.get_filter_params(user, "test_rule", rule_context)
    assert params is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
