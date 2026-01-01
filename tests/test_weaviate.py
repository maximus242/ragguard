"""
Tests for Weaviate integration.
"""

from unittest.mock import MagicMock, Mock

import pytest


def test_weaviate_filter_builder_basic():
    """Test basic Weaviate filter building."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'public_docs',
                'match': {'visibility': 'public'},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice'}

    filter_dict = to_weaviate_filter(policy, user)

    assert filter_dict is not None
    assert filter_dict['path'] == ['visibility']
    assert filter_dict['operator'] == 'Equal'
    assert filter_dict['valueText'] == 'public'


def test_weaviate_filter_builder_multiple_rules():
    """Test Weaviate filter with multiple rules (OR logic)."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'public',
                'match': {'visibility': 'public'},
                'allow': {'everyone': True}
            },
            {
                'name': 'engineering',
                'match': {'department': 'engineering'},
                'allow': {'roles': ['engineer']}
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'roles': ['engineer']}

    filter_dict = to_weaviate_filter(policy, user)

    # Should have OR logic between rules
    assert filter_dict is not None
    assert filter_dict['operator'] == 'Or'
    assert len(filter_dict['operands']) == 2


def test_weaviate_filter_builder_list_values():
    """Test Weaviate filter with match conditions containing list values."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'multiple_statuses',
                'match': {'status': ['active', 'pending', 'review']},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice'}

    filter_dict = to_weaviate_filter(policy, user)

    # Should create ContainsAny for list values
    assert filter_dict is not None
    assert filter_dict['path'] == ['status']
    assert filter_dict['operator'] == 'ContainsAny'
    assert filter_dict['valueText'] == ['active', 'pending', 'review']


def test_weaviate_filter_builder_condition_equal():
    """Test Weaviate filter with equality condition."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'department_match',
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'department': 'engineering'}

    filter_dict = to_weaviate_filter(policy, user)

    # Should create Equal condition
    assert filter_dict is not None
    assert filter_dict['path'] == ['department']
    assert filter_dict['operator'] == 'Equal'
    assert filter_dict['valueText'] == 'engineering'


def test_weaviate_filter_builder_condition_in():
    """Test Weaviate filter with 'in' condition."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'shared_docs',
                'allow': {
                    'conditions': ['user.id in document.shared_with']
                }
            }
        ],
        'default': 'deny'
    })

    user = {'id': 'alice@example.com'}

    filter_dict = to_weaviate_filter(policy, user)

    # Should create ContainsAny condition
    assert filter_dict is not None
    assert filter_dict['path'] == ['shared_with']
    assert filter_dict['operator'] == 'ContainsAny'
    assert filter_dict['valueText'] == ['alice@example.com']


def test_weaviate_filter_builder_default_allow():
    """Test Weaviate filter with default allow and no matching rules."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {'name': 'admin_only', 'allow': {'roles': ['admin']}}
        ],
        'default': 'allow'
    })

    user = {'id': 'alice', 'roles': ['user']}

    filter_dict = to_weaviate_filter(policy, user)

    # Should return None (allow all)
    assert filter_dict is None


def test_weaviate_filter_builder_default_deny():
    """Test Weaviate filter with default deny and no matching rules."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {'name': 'admin_only', 'allow': {'roles': ['admin']}}
        ],
        'default': 'deny'
    })

    user = {'id': 'alice', 'roles': ['user']}

    filter_dict = to_weaviate_filter(policy, user)

    # Should return deny-all filter using internal constant
    from ragguard.constants import DENY_ALL_FIELD
    assert filter_dict is not None
    assert filter_dict['path'] == [DENY_ALL_FIELD]
    assert filter_dict['operator'] == 'Equal'


def test_weaviate_filter_builder_value_types():
    """Test Weaviate filter with different value types."""
    from ragguard import Policy
    from ragguard.filters.builder import to_weaviate_filter

    # Test integer
    policy_int = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'test',
                'match': {'priority': 1},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    filter_dict = to_weaviate_filter(policy_int, {'id': 'alice'})
    assert filter_dict['valueInt'] == 1

    # Test boolean
    policy_bool = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'test',
                'match': {'is_public': True},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    filter_dict = to_weaviate_filter(policy_bool, {'id': 'alice'})
    assert filter_dict['valueBoolean'] is True

    # Test float
    policy_float = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'test',
                'match': {'score': 0.5},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    filter_dict = to_weaviate_filter(policy_float, {'id': 'alice'})
    assert filter_dict['valueNumber'] == 0.5


def test_weaviate_retriever_initialization():
    """Test WeaviateSecureRetriever initialization."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    # Create a mock Weaviate client
    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    assert retriever.collection == 'Documents'
    assert retriever.backend_name == 'weaviate'


def test_weaviate_retriever_wrong_client_type():
    """Test WeaviateSecureRetriever raises error with wrong client type."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever
    from ragguard.exceptions import RetrieverError

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    # Pass a wrong client type
    wrong_client = Mock()
    wrong_client.__class__.__name__ = 'QdrantClient'

    with pytest.raises(RetrieverError, match="Expected weaviate.Client"):
        WeaviateSecureRetriever(
            client=wrong_client,
            collection='Documents',
            policy=policy
        )


def test_weaviate_retriever_missing_dependency():
    """Test WeaviateSecureRetriever raises error when weaviate-client not installed."""
    import builtins

    from ragguard import Policy
    from ragguard.exceptions import RetrieverError

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    # Mock import to fail
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'weaviate':
            raise ImportError("No module named 'weaviate'")
        return real_import(name, *args, **kwargs)

    from unittest.mock import patch
    mock_client = Mock()

    with patch.object(builtins, '__import__', side_effect=mock_import):
        with pytest.raises(RetrieverError, match="weaviate-client not installed"):
            from ragguard.retrievers.weaviate import WeaviateSecureRetriever
            retriever = object.__new__(WeaviateSecureRetriever)
            retriever.__init__(
                client=mock_client,
                collection='Documents',
                policy=policy
            )


def test_weaviate_retriever_search():
    """Test WeaviateSecureRetriever search execution."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    # Create a mock Weaviate client
    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    # Mock the query builder chain
    mock_query_result = {
        'data': {
            'Get': {
                'Documents': [
                    {
                        'text': 'Document 1',
                        '_additional': {'id': '1', 'certainty': 0.9}
                    },
                    {
                        'text': 'Document 2',
                        '_additional': {'id': '2', 'certainty': 0.85}
                    }
                ]
            }
        }
    }

    # Set up the mock chain
    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = mock_query_result

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'public',
                'match': {'visibility': 'public'},
                'allow': {'everyone': True}
            }
        ],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    user = {'id': 'alice'}

    # Execute search
    results = retriever.search([0.1, 0.2, 0.3], user=user, limit=5)

    # Verify results
    assert len(results) == 2
    assert results[0]['text'] == 'Document 1'
    assert results[1]['text'] == 'Document 2'

    # Verify query was built correctly
    mock_get.assert_called_once_with('Documents', ['*'])
    mock_builder.with_near_vector.assert_called_once()
    mock_builder.with_limit.assert_called_once_with(5)
    mock_builder.with_where.assert_called_once()


def test_weaviate_retriever_search_with_custom_properties():
    """Test WeaviateSecureRetriever with custom properties."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    mock_query_result = {
        'data': {
            'Get': {
                'Documents': [
                    {'title': 'Doc 1', 'content': 'Content 1'}
                ]
            }
        }
    }

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = mock_query_result

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    # Search with custom properties
    results = retriever.search(
        [0.1, 0.2, 0.3],
        user={'id': 'alice'},
        limit=5,
        properties=['title', 'content']
    )

    # Verify custom properties were passed
    mock_get.assert_called_once_with('Documents', ['title', 'content'])
    assert len(results) == 1


def test_weaviate_retriever_search_with_certainty():
    """Test WeaviateSecureRetriever with custom certainty."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = {'data': {'Get': {'Documents': []}}}

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    # Search with custom certainty
    retriever.search(
        [0.1, 0.2, 0.3],
        user={'id': 'alice'},
        certainty=0.85
    )

    # Verify certainty was passed
    call_args = mock_builder.with_near_vector.call_args
    assert call_args[0][0]['certainty'] == 0.85


def test_weaviate_retriever_search_with_additional():
    """Test WeaviateSecureRetriever with custom additional fields."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = {'data': {'Get': {'Documents': []}}}

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    # Search with custom additional fields
    retriever.search(
        [0.1, 0.2, 0.3],
        user={'id': 'alice'},
        with_additional=['distance', 'score']
    )

    # Verify additional fields were passed
    mock_builder.with_additional.assert_called_once_with(['distance', 'score'])


def test_weaviate_retriever_search_empty_results():
    """Test WeaviateSecureRetriever with no results."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    # Empty result
    mock_query_result = {
        'data': {
            'Get': {
                'Documents': []
            }
        }
    }

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = mock_query_result

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    results = retriever.search([0.1, 0.2, 0.3], user={'id': 'alice'})

    assert len(results) == 0


def test_weaviate_retriever_search_malformed_response():
    """Test WeaviateSecureRetriever handles malformed responses."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    # Malformed result (missing 'Get' key)
    mock_query_result = {
        'data': {}
    }

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = mock_query_result

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    results = retriever.search([0.1, 0.2, 0.3], user={'id': 'alice'})

    # Should return empty list instead of crashing
    assert len(results) == 0


def test_weaviate_retriever_search_error_handling():
    """Test WeaviateSecureRetriever handles search errors."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever
    from ragguard.exceptions import RetrieverError

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    # Mock search to raise exception
    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.side_effect = Exception("Weaviate connection failed")

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    # Should raise RetrieverError
    with pytest.raises(RetrieverError, match="Weaviate search failed"):
        retriever.search([0.1, 0.2, 0.3], user={'id': 'alice'})


def test_weaviate_retriever_with_embed_fn():
    """Test WeaviateSecureRetriever with string query and embed_fn."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'Client'

    mock_query_result = {
        'data': {
            'Get': {
                'Documents': [{'text': 'Result'}]
            }
        }
    }

    mock_builder = Mock()
    mock_builder.with_near_vector.return_value = mock_builder
    mock_builder.with_limit.return_value = mock_builder
    mock_builder.with_where.return_value = mock_builder
    mock_builder.with_additional.return_value = mock_builder
    mock_builder.do.return_value = mock_query_result

    mock_get = Mock(return_value=mock_builder)
    mock_client.query.get = mock_get

    # Mock embedding function
    def mock_embed(text):
        return [0.1, 0.2, 0.3]

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy,
        embed_fn=mock_embed
    )

    # Search with string query
    results = retriever.search("test query", user={'id': 'alice'})

    assert len(results) == 1
    # Verify embedding was called
    call_args = mock_builder.with_near_vector.call_args
    assert call_args[0][0]['vector'] == [0.1, 0.2, 0.3]


def test_weaviate_retriever_weaviate_client_v4():
    """Test WeaviateSecureRetriever accepts WeaviateClient (v4 API)."""
    pytest.importorskip("weaviate")

    from ragguard import Policy, WeaviateSecureRetriever

    mock_client = Mock()
    mock_client.__class__.__name__ = 'WeaviateClient'  # v4 client name

    policy = Policy.from_dict({
        'version': '1',
        'rules': [{'name': 'allow_all', 'allow': {'everyone': True}}],
        'default': 'deny'
    })

    # Should not raise error for WeaviateClient
    retriever = WeaviateSecureRetriever(
        client=mock_client,
        collection='Documents',
        policy=policy
    )

    assert retriever.collection == 'Documents'
    assert retriever.backend_name == 'weaviate'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
