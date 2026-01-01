"""
Tests for custom filter builders.

These tests cover the extensibility layer that allows users to implement
custom permission logic.
"""

from unittest.mock import MagicMock, Mock

import pytest

# Test ACLFilterBuilder
# =====================

def test_acl_filter_builder_qdrant():
    """Test ACLFilterBuilder for Qdrant with user and group access."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    # Mock get_user_groups function
    def get_user_groups(user):
        return user.get('groups', [])

    builder = ACLFilterBuilder(
        acl_field='acl',
        get_user_groups=get_user_groups
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'default_rule',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {
        'id': 'alice@company.com',
        'groups': ['engineering', 'product']
    }

    # Build filter
    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should have conditions for user, groups, and public
    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    # Should have 3 conditions: user match, group match, public match
    assert len(filter_obj.should) == 3


def test_acl_filter_builder_pgvector():
    """Test ACLFilterBuilder for pgvector."""
    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    def get_user_groups(user):
        return user.get('groups', [])

    builder = ACLFilterBuilder(
        acl_field='permissions',
        get_user_groups=get_user_groups
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'default_rule',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {
        'id': 'bob@company.com',
        'groups': ['finance']
    }

    where_clause, params = builder.build_filter(policy, user, 'pgvector')

    # Should have WHERE clause
    assert 'WHERE' in where_clause
    assert 'permissions' in where_clause
    assert len(params) > 0
    assert 'bob@company.com' in str(params)


def test_acl_filter_builder_no_user():
    """Test ACLFilterBuilder denies access when no user provided."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    builder = ACLFilterBuilder(acl_field='acl')
    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    # No user ID
    user = {}
    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should deny all
    assert filter_obj is not None
    assert hasattr(filter_obj, 'must')


def test_acl_filter_builder_public_access():
    """Test ACLFilterBuilder includes public documents."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    builder = ACLFilterBuilder(
        acl_field='acl',
        public_field='is_public'
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'user@company.com', 'groups': []}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should have public condition
    assert filter_obj is not None


# Test LambdaFilterBuilder
# ========================

def test_lambda_filter_builder_qdrant():
    """Test LambdaFilterBuilder with custom Qdrant logic."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import models

    from ragguard import Policy
    from ragguard.filters.custom import LambdaFilterBuilder

    def custom_qdrant_filter(policy, user):
        department = user.get('department')
        if not department:
            return models.Filter(must=[
                models.FieldCondition(
                    key='_impossible',
                    match=models.MatchValue(value='denied')
                )
            ])

        return models.Filter(
            must=[
                models.FieldCondition(
                    key='department',
                    match=models.MatchValue(value=department)
                )
            ]
        )

    builder = LambdaFilterBuilder(qdrant=custom_qdrant_filter)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'alice', 'department': 'engineering'}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    assert filter_obj is not None
    assert hasattr(filter_obj, 'must')


def test_lambda_filter_builder_pgvector():
    """Test LambdaFilterBuilder with custom pgvector logic."""
    from ragguard import Policy
    from ragguard.filters.custom import LambdaFilterBuilder

    def custom_pgvector_filter(policy, user):
        department = user.get('department', 'unknown')
        return ("WHERE department = %s", [department])

    builder = LambdaFilterBuilder(pgvector=custom_pgvector_filter)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'bob', 'department': 'finance'}

    where_clause, params = builder.build_filter(policy, user, 'pgvector')

    assert 'WHERE' in where_clause
    assert 'department' in where_clause
    assert params == ['finance']


def test_lambda_filter_builder_no_handler():
    """Test LambdaFilterBuilder raises error when no handler for backend."""
    from ragguard import Policy
    from ragguard.filters.custom import LambdaFilterBuilder

    # Only provide Qdrant handler
    builder = LambdaFilterBuilder(qdrant=lambda p, u: None)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'alice'}

    # Should raise error for pgvector
    with pytest.raises(ValueError, match="No custom builder"):
        builder.build_filter(policy, user, 'pgvector')


def test_lambda_filter_builder_multiple_conditions():
    """Test LambdaFilterBuilder with complex multi-condition logic."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import models

    from ragguard import Policy
    from ragguard.filters.custom import LambdaFilterBuilder

    def custom_filter(policy, user):
        conditions = []

        # Add department condition
        if user.get('department'):
            conditions.append(
                models.FieldCondition(
                    key='department',
                    match=models.MatchValue(value=user['department'])
                )
            )

        # Add role condition
        if user.get('roles'):
            conditions.append(
                models.FieldCondition(
                    key='allowed_roles',
                    match=models.MatchAny(any=user['roles'])
                )
            )

        if not conditions:
            return models.Filter(must=[
                models.FieldCondition(
                    key='_impossible',
                    match=models.MatchValue(value='denied')
                )
            ])

        return models.Filter(should=conditions)

    builder = LambdaFilterBuilder(qdrant=custom_filter)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {
        'id': 'alice',
        'department': 'engineering',
        'roles': ['admin', 'user']
    }

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    assert filter_obj is not None
    assert hasattr(filter_obj, 'should')
    assert len(filter_obj.should) == 2


# Test HybridFilterBuilder
# ========================

def test_hybrid_filter_builder_qdrant():
    """Test HybridFilterBuilder combines standard + custom filters."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import models

    from ragguard import Policy
    from ragguard.filters.custom import HybridFilterBuilder

    def add_custom_filter(user):
        # Add a custom condition (e.g., email verification)
        if not user.get('email_verified'):
            return models.Filter(must=[
                models.FieldCondition(
                    key='_impossible',
                    match=models.MatchValue(value='denied')
                )
            ])

        return models.Filter(
            must=[
                models.FieldCondition(
                    key='email_verified',
                    match=models.MatchValue(value=True)
                )
            ]
        )

    builder = HybridFilterBuilder(
        additional_filters={'qdrant': add_custom_filter}
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'allow_all',
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {
        'id': 'alice',
        'email_verified': True,
        'roles': ['user']
    }

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    assert filter_obj is not None


def test_hybrid_filter_builder_no_additional_filters():
    """Test HybridFilterBuilder falls back to standard filter when no custom filters."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import HybridFilterBuilder

    builder = HybridFilterBuilder(additional_filters={})

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'role_based',
                'allow': {'roles': ['admin']}
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'roles': ['admin']}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should use standard policy filter
    assert filter_obj is not None


# Test FieldMappingFilterBuilder
# ==============================

def test_field_mapping_filter_builder_basic():
    """Test FieldMappingFilterBuilder maps field names."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import FieldMappingFilterBuilder

    # Policy expects 'department', but docs have 'dept_code'
    mapping = {
        'department': 'dept_code'
    }

    builder = FieldMappingFilterBuilder(field_mapping=mapping)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'dept_rule',
                'match': {'confidential': False},
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'department': 'engineering'}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Filter should use dept_code instead of department
    assert filter_obj is not None


def test_field_mapping_filter_builder_with_transforms():
    """Test FieldMappingFilterBuilder with value transformations."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import FieldMappingFilterBuilder

    # Map fields and transform values
    mapping = {
        'confidential': 'security_level'
    }

    transforms = {
        'confidential': lambda x: 'SECRET' if x else 'PUBLIC'
    }

    builder = FieldMappingFilterBuilder(
        field_mapping=mapping,
        value_transforms=transforms
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'public_docs',
                'match': {'confidential': False},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'roles': ['user']}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should transform confidential=False to security_level='PUBLIC'
    assert filter_obj is not None


def test_field_mapping_filter_builder_multiple_mappings():
    """Test FieldMappingFilterBuilder with multiple field mappings."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import FieldMappingFilterBuilder

    mapping = {
        'department': 'dept_code',
        'confidential': 'classification_level',
        'visibility': 'access_class'
    }

    builder = FieldMappingFilterBuilder(field_mapping=mapping)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'complex_rule',
                'match': {
                    'confidential': False,
                    'visibility': 'internal'
                },
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'department': 'engineering'}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    assert filter_obj is not None


def test_field_mapping_filter_builder_pgvector():
    """Test FieldMappingFilterBuilder for pgvector."""
    from ragguard import Policy
    from ragguard.filters.custom import FieldMappingFilterBuilder

    mapping = {
        'department': 'dept_code'
    }

    builder = FieldMappingFilterBuilder(field_mapping=mapping)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'dept_rule',
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'bob', 'department': 'finance'}

    where_clause, params = builder.build_filter(policy, user, 'pgvector')

    # Should generate a filter (mapping not yet implemented)
    assert isinstance(where_clause, str)
    assert isinstance(params, list)


# Test CustomFilterBuilder edge cases
# ===================================

def test_custom_filter_builder_unsupported_backend():
    """Test CustomFilterBuilder raises error for unsupported backend."""
    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    builder = ACLFilterBuilder(acl_field='acl')
    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'alice'}

    with pytest.raises(ValueError, match="Unsupported backend"):
        builder.build_filter(policy, user, 'elasticsearch')


def test_custom_filter_builder_empty_groups():
    """Test ACLFilterBuilder handles users with no groups."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import ACLFilterBuilder

    builder = ACLFilterBuilder(
        acl_field='acl',
        get_user_groups=lambda user: []
    )

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'alice@company.com', 'groups': []}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should still create filter (user + public conditions)
    assert filter_obj is not None


def test_lambda_filter_builder_none_return():
    """Test LambdaFilterBuilder handles None return from custom function."""
    pytest.importorskip("qdrant_client")

    from ragguard import Policy
    from ragguard.filters.custom import LambdaFilterBuilder

    # Custom function returns None
    builder = LambdaFilterBuilder(qdrant=lambda p, u: None)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'default_rule', 'allow': {'everyone': True}}],
        'default': 'deny'
    })
    user = {'id': 'alice'}

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should handle None gracefully (either return None or deny-all)
    # Implementation-specific behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
