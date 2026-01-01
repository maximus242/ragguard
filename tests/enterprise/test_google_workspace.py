"""
Tests for Google Workspace integration (Enterprise).

These tests use mocking to avoid requiring actual Google Workspace credentials.

Note: Google Workspace integration moved to ragguard-enterprise.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests - requires ragguard-enterprise package
pytestmark = pytest.mark.skip(reason="Requires ragguard-enterprise package - not part of open-source version")


# Test GoogleWorkspaceResolver
# ============================

def test_google_workspace_resolver_get_user_groups():
    """Test getting user groups from Google Workspace."""
    try:
        from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver
    except ImportError:
        pytest.skip("GoogleWorkspaceResolver requires ragguard-enterprise")

    # Create mock credentials
    mock_creds = Mock()

    # Create resolver
    resolver = GoogleWorkspaceResolver(
        credentials=mock_creds,
        cache_ttl=300,
        domain="company.com"
    )

    # Mock the directory service
    mock_directory_service = Mock()
    mock_groups_list = Mock()
    mock_directory_service.groups.return_value.list.return_value = mock_groups_list

    # Mock response
    mock_groups_list.execute.return_value = {
        'groups': [
            {'email': 'engineering@company.com'},
            {'email': 'product@company.com'}
        ]
    }

    resolver._directory_service = mock_directory_service

    # Test
    groups = resolver.get_user_groups('alice@company.com')

    assert groups == ['engineering@company.com', 'product@company.com']
    mock_groups_list.execute.assert_called_once()


def test_google_workspace_resolver_get_user_groups_pagination():
    """Test getting user groups with pagination."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock directory service with pagination
    mock_directory_service = Mock()
    mock_groups_list = Mock()
    mock_directory_service.groups.return_value.list.return_value = mock_groups_list

    # Mock paginated responses
    mock_groups_list.execute.side_effect = [
        {
            'groups': [
                {'email': 'group1@company.com'},
                {'email': 'group2@company.com'}
            ],
            'nextPageToken': 'token123'
        },
        {
            'groups': [
                {'email': 'group3@company.com'}
            ]
        }
    ]

    resolver._directory_service = mock_directory_service

    # Test
    groups = resolver.get_user_groups('alice@company.com')

    assert len(groups) == 3
    assert 'group1@company.com' in groups
    assert 'group2@company.com' in groups
    assert 'group3@company.com' in groups
    assert mock_groups_list.execute.call_count == 2


def test_google_workspace_resolver_get_user_info():
    """Test getting user info from Google Workspace."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock directory service
    mock_directory_service = Mock()
    mock_users_get = Mock()
    mock_directory_service.users.return_value.get.return_value = mock_users_get

    # Mock response
    mock_users_get.execute.return_value = {
        'primaryEmail': 'alice@company.com',
        'name': {'fullName': 'Alice Smith'},
        'orgUnitPath': '/Engineering',
        'organizations': [{'department': 'Backend', 'title': 'Senior Engineer'}],
        'suspended': False
    }

    resolver._directory_service = mock_directory_service

    # Test
    user_info = resolver.get_user_info('alice@company.com')

    assert user_info['email'] == 'alice@company.com'
    assert user_info['name'] == 'Alice Smith'
    assert user_info['orgUnitPath'] == '/Engineering'
    assert user_info['department'] == 'Backend'
    assert user_info['title'] == 'Senior Engineer'
    assert user_info['suspended'] is False


def test_google_workspace_resolver_check_file_permission_direct():
    """Test checking file permission with direct user access."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock drive service
    mock_drive_service = Mock()
    mock_permissions_list = Mock()
    mock_drive_service.permissions.return_value.list.return_value = mock_permissions_list

    # Mock response with direct user permission
    mock_permissions_list.execute.return_value = {
        'permissions': [
            {
                'emailAddress': 'alice@company.com',
                'type': 'user',
                'role': 'reader'
            }
        ]
    }

    resolver._drive_service = mock_drive_service

    # Mock get_user_groups to avoid additional calls
    resolver.get_user_groups = Mock(return_value=[])

    # Test
    has_access = resolver.check_file_permission('file123', 'alice@company.com')

    assert has_access is True


def test_google_workspace_resolver_check_file_permission_group():
    """Test checking file permission with group access."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock drive service
    mock_drive_service = Mock()
    mock_permissions_list = Mock()
    mock_drive_service.permissions.return_value.list.return_value = mock_permissions_list

    # Mock response with group permission
    mock_permissions_list.execute.return_value = {
        'permissions': [
            {
                'emailAddress': 'engineering@company.com',
                'type': 'group',
                'role': 'reader'
            }
        ]
    }

    resolver._drive_service = mock_drive_service

    # Mock get_user_groups
    resolver.get_user_groups = Mock(return_value=['engineering@company.com', 'product@company.com'])

    # Test
    has_access = resolver.check_file_permission('file123', 'alice@company.com')

    assert has_access is True


def test_google_workspace_resolver_check_file_permission_domain():
    """Test checking file permission with domain-wide access."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock drive service
    mock_drive_service = Mock()
    mock_permissions_list = Mock()
    mock_drive_service.permissions.return_value.list.return_value = mock_permissions_list

    # Mock response with domain permission
    mock_permissions_list.execute.return_value = {
        'permissions': [
            {
                'type': 'domain',
                'role': 'reader'
            }
        ]
    }

    resolver._drive_service = mock_drive_service
    resolver.get_user_groups = Mock(return_value=[])

    # Test - user in domain
    has_access = resolver.check_file_permission('file123', 'alice@company.com')
    assert has_access is True

    # Test - user not in domain
    has_access = resolver.check_file_permission('file123', 'external@other.com')
    assert has_access is False


def test_google_workspace_resolver_check_file_permission_anyone():
    """Test checking file permission with public access."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock drive service
    mock_drive_service = Mock()
    mock_permissions_list = Mock()
    mock_drive_service.permissions.return_value.list.return_value = mock_permissions_list

    # Mock response with anyone permission
    mock_permissions_list.execute.return_value = {
        'permissions': [
            {
                'type': 'anyone',
                'role': 'reader'
            }
        ]
    }

    resolver._drive_service = mock_drive_service
    resolver.get_user_groups = Mock(return_value=[])

    # Test
    has_access = resolver.check_file_permission('file123', 'anyone@anywhere.com')

    assert has_access is True


def test_google_workspace_resolver_caching():
    """Test that results are properly cached."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, cache_ttl=300)

    # Mock directory service
    mock_directory_service = Mock()
    mock_groups_list = Mock()
    mock_directory_service.groups.return_value.list.return_value = mock_groups_list

    mock_groups_list.execute.return_value = {
        'groups': [{'email': 'test@company.com'}]
    }

    resolver._directory_service = mock_directory_service

    # First call
    groups1 = resolver.get_user_groups('alice@company.com')

    # Second call should use cache
    groups2 = resolver.get_user_groups('alice@company.com')

    assert groups1 == groups2
    # Should only call execute once due to caching
    mock_groups_list.execute.assert_called_once()


def test_google_workspace_resolver_can_access():
    """Test can_access method."""
    from ragguard_enterprise.integrations.google_workspace import GoogleWorkspaceResolver

    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock methods
    resolver.get_user_info = Mock(return_value={
        'email': 'alice@company.com',
        'suspended': False
    })
    resolver.get_user_groups = Mock(return_value=['engineering@company.com'])

    # Test - user with required group
    user = {'id': 'alice@company.com'}
    context = {'required_groups': ['engineering@company.com']}
    assert resolver.can_access(user, 'test_rule', context) is True

    # Test - user without required group
    context = {'required_groups': ['finance@company.com']}
    assert resolver.can_access(user, 'test_rule', context) is False

    # Test - suspended user
    resolver.get_user_info = Mock(return_value={
        'email': 'alice@company.com',
        'suspended': True
    })
    context = {'required_groups': ['engineering@company.com']}
    assert resolver.can_access(user, 'test_rule', context) is False


# Test GoogleDrivePermissionBuilder
# =================================

def test_google_drive_permission_builder_qdrant():
    """Test GoogleDrivePermissionBuilder for Qdrant."""
    from ragguard_enterprise.integrations.google_workspace import (
        GoogleDrivePermissionBuilder,
        GoogleWorkspaceResolver,
    )

    # Create mock resolver
    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")
    resolver.get_filter_params = Mock(return_value={
        'groups': ['engineering@company.com', 'product@company.com'],
        'org_unit': '/Engineering',
        'email': 'alice@company.com'
    })

    # Create builder
    builder = GoogleDrivePermissionBuilder(
        workspace_resolver=resolver,
        file_id_field='drive_file_id'
    )

    # Test
    user = {'id': 'alice@company.com'}
    policy = Mock()

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should have conditions for user, groups, domain, and public
    assert filter_obj is not None
    # Filter should be a Qdrant Filter with should conditions
    assert hasattr(filter_obj, 'should')


def test_google_drive_permission_builder_pgvector():
    """Test GoogleDrivePermissionBuilder for pgvector."""
    from ragguard_enterprise.integrations.google_workspace import (
        GoogleDrivePermissionBuilder,
        GoogleWorkspaceResolver,
    )

    # Create mock resolver
    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")
    resolver.get_filter_params = Mock(return_value={
        'groups': ['engineering@company.com'],
        'org_unit': '/Engineering',
        'email': 'alice@company.com'
    })

    # Create builder
    builder = GoogleDrivePermissionBuilder(
        workspace_resolver=resolver,
        file_id_field='drive_file_id'
    )

    # Test
    user = {'id': 'alice@company.com'}
    policy = Mock()

    where_clause, params = builder.build_filter(policy, user, 'pgvector')

    # Should have WHERE clause with multiple conditions
    assert 'WHERE' in where_clause
    assert 'drive_permissions' in where_clause
    assert len(params) > 0


def test_google_drive_permission_builder_no_access():
    """Test GoogleDrivePermissionBuilder denies access when user has no permissions."""
    from ragguard_enterprise.integrations.google_workspace import (
        GoogleDrivePermissionBuilder,
        GoogleWorkspaceResolver,
    )

    # Create mock resolver that returns no permissions
    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")
    resolver.get_filter_params = Mock(return_value={})  # No permissions

    # Create builder
    builder = GoogleDrivePermissionBuilder(workspace_resolver=resolver)

    # Test Qdrant
    user = {'id': 'alice@company.com'}
    policy = Mock()

    filter_obj = builder.build_filter(policy, user, 'qdrant')

    # Should return deny-all filter
    assert filter_obj is not None
    assert hasattr(filter_obj, 'must')


# Test GoogleCloudIAMResolver
# ===========================

def test_google_cloud_iam_resolver_check_permission():
    """Test checking IAM permissions."""
    from ragguard_enterprise.integrations.google_workspace import GoogleCloudIAMResolver

    mock_creds = Mock()
    resolver = GoogleCloudIAMResolver(
        credentials=mock_creds,
        project_id="test-project"
    )

    # Mock IAM service
    mock_iam_service = Mock()
    mock_test_permissions = Mock()
    mock_iam_service.projects.return_value.testIamPermissions.return_value = mock_test_permissions

    # Mock response
    mock_test_permissions.execute.return_value = {
        'permissions': ['storage.buckets.get', 'storage.buckets.list']
    }

    resolver._iam_service = mock_iam_service

    # Test - user has all required permissions
    has_access = resolver.check_iam_permission(
        resource='projects/test-project',
        email='alice@company.com',
        permissions=['storage.buckets.get', 'storage.buckets.list']
    )
    assert has_access is True

    # Test - user missing some permissions
    has_access = resolver.check_iam_permission(
        resource='projects/test-project',
        email='alice@company.com',
        permissions=['storage.buckets.get', 'storage.buckets.delete']
    )
    assert has_access is False


def test_google_cloud_iam_resolver_can_access():
    """Test can_access method."""
    from ragguard_enterprise.integrations.google_workspace import GoogleCloudIAMResolver

    mock_creds = Mock()
    resolver = GoogleCloudIAMResolver(
        credentials=mock_creds,
        project_id="test-project"
    )

    # Mock check_iam_permission
    resolver.check_iam_permission = Mock(return_value=True)

    # Test
    user = {'id': 'alice@company.com'}
    context = {
        'gcp_resource': 'projects/test-project',
        'required_permissions': ['storage.buckets.get']
    }

    assert resolver.can_access(user, 'test_rule', context) is True

    # Verify the method was called correctly
    resolver.check_iam_permission.assert_called_once_with(
        'projects/test-project',
        'alice@company.com',
        ['storage.buckets.get']
    )


# Integration Tests
# ================

def test_google_workspace_integration_with_retriever():
    """Test Google Workspace integration with SecureRetriever."""
    pytest.importorskip("qdrant_client")

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from ragguard_enterprise.integrations.google_workspace import (
        GoogleDrivePermissionBuilder,
        GoogleWorkspaceResolver,
    )

    from ragguard import Policy, SecureRetriever

    # Create mock resolver
    mock_creds = Mock()
    resolver = GoogleWorkspaceResolver(credentials=mock_creds, domain="company.com")

    # Mock resolver methods
    resolver.get_filter_params = Mock(return_value={
        'groups': ['engineering@company.com'],
        'org_unit': '/Engineering',
        'email': 'alice@company.com'
    })

    # Create filter builder
    builder = GoogleDrivePermissionBuilder(workspace_resolver=resolver)

    # Create policy
    policy = Policy.from_dict({
        'version': '1',
        'default': 'deny',
        'rules': [{'name': 'drive', 'allow': {}}]
    })

    # Initialize Qdrant
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_docs",
        vectors_config=VectorParams(size=3, distance=Distance.COSINE)
    )

    # Index test documents
    client.upsert(
        collection_name="test_docs",
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3],
                payload={
                    "text": "Test doc 1",
                    "drive_permissions": {
                        "users": ["alice@company.com"],
                        "groups": [],
                        "domain": None,
                        "anyone": False
                    }
                }
            ),
            PointStruct(
                id=2,
                vector=[0.2, 0.3, 0.4],
                payload={
                    "text": "Test doc 2",
                    "drive_permissions": {
                        "users": [],
                        "groups": ["engineering@company.com"],
                        "domain": None,
                        "anyone": False
                    }
                }
            )
        ]
    )

    # Create retriever
    retriever = SecureRetriever(
        client=client,
        collection="test_docs",
        policy=policy,
        embed_fn=lambda text: [0.1, 0.2, 0.3],
        custom_filter_builder=builder
    )

    # Test search
    user = {"id": "alice@company.com"}
    results = retriever.search("test", user=user, limit=10)

    # Should find both documents (alice has direct access to doc1, group access to doc2)
    assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
