"""
Tests for FilterBuilderBase.

Verifies that the new base class produces identical output to the
original backend-specific implementations.
"""

import pytest

from ragguard.filters.backends.chromadb import to_chromadb_filter
from ragguard.filters.backends.pinecone import to_pinecone_filter
from ragguard.filters.builder_base import (
    ChromaDBFilterBuilder,
    DictFilterBuilder,
    FilterBuilderBase,
    PineconeFilterBuilder,
    to_chromadb_filter_v2,
    to_pinecone_filter_v2,
)
from ragguard.policy.models import AllowConditions, Policy, Rule


class TestPineconeFilterBuilderEquivalence:
    """Test that PineconeFilterBuilder produces same output as to_pinecone_filter."""

    def test_simple_match(self):
        """Test simple field match."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="public",
                    match={"type": "public"},
                    allow=AllowConditions(everyone=True),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_list_match(self):
        """Test list match with $in operator."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="shared",
                    match={"type": ["public", "shared"]},
                    allow=AllowConditions(everyone=True),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_user_equals_document(self):
        """Test user.field == document.field condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="dept_match",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["user.department == document.department"],
                    ),
                )
            ],
        )
        user = {"id": "alice", "department": "engineering"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_user_in_document_array(self):
        """Test user.id in document.allowed_users condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="allowed_users",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["user.id in document.allowed_users"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_field_in_list(self):
        """Test document.field in ['a', 'b', 'c'] condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="category_filter",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.category in ['public', 'shared']"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_not_in(self):
        """Test document.field not in ['a', 'b'] condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="exclude_categories",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.category not in ['secret', 'classified']"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_exists(self):
        """Test document.field exists condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="published",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.published_at exists"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_not_exists(self):
        """Test document.field not exists condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="not_deleted",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.deleted_at not exists"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_not_equals(self):
        """Test document.field != value condition."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="active",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.status != 'deleted'"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_multiple_rules_or(self):
        """Test multiple rules combined with OR."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="public",
                    match={"type": "public"},
                    allow=AllowConditions(everyone=True),
                ),
                Rule(
                    name="dept_match",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["user.department == document.department"],
                    ),
                ),
            ],
        )
        user = {"id": "alice", "department": "engineering"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_multiple_conditions_and(self):
        """Test multiple conditions combined with AND."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="internal_dept",
                    match={"type": "internal"},
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["user.department == document.department"],
                    ),
                )
            ],
        )
        user = {"id": "alice", "department": "engineering"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_deny_all_when_user_field_none(self):
        """Test deny all when user field is None."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="dept_match",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["user.department == document.department"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}  # No department

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_deny_all_default(self):
        """Test deny all when no rules match and default is deny."""
        policy = Policy(
            version="1",
            default="deny",
            rules=[
                Rule(
                    name="admin_only",
                    allow=AllowConditions(roles=["admin"]),
                )
            ],
        )
        user = {"id": "alice", "roles": ["user"]}  # Not admin

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new

    def test_allow_all_default(self):
        """Test allow all when no rules match and default is allow."""
        policy = Policy(
            version="1",
            default="allow",
            rules=[
                Rule(
                    name="admin_only",
                    allow=AllowConditions(roles=["admin"]),
                )
            ],
        )
        user = {"id": "alice", "roles": ["user"]}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new  # Both should be None


class TestChromaDBFilterBuilderEquivalence:
    """Test that ChromaDBFilterBuilder produces same output as to_chromadb_filter."""

    def test_simple_match(self):
        """Test simple field match."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="public",
                    match={"type": "public"},
                    allow=AllowConditions(everyone=True),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_chromadb_filter(policy, user)
        new = to_chromadb_filter_v2(policy, user)

        assert original == new

    def test_exists_uses_none(self):
        """Test that ChromaDB uses $ne None for exists."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="published",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.published_at exists"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_chromadb_filter(policy, user)
        new = to_chromadb_filter_v2(policy, user)

        assert original == new
        # Verify it uses $ne None, not $exists
        assert "$exists" not in str(original)
        assert "$ne" in str(original) or "None" in str(original)

    def test_complex_policy(self):
        """Test complex policy with multiple rules and conditions."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="public",
                    match={"type": "public"},
                    allow=AllowConditions(everyone=True),
                ),
                Rule(
                    name="admin",
                    allow=AllowConditions(
                        roles=["admin"],
                    ),
                ),
                Rule(
                    name="dept_match",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=[
                            "user.department == document.department",
                            "document.status != 'draft'",
                        ],
                    ),
                ),
            ],
        )
        user = {"id": "alice", "department": "engineering", "roles": ["user"]}

        original = to_chromadb_filter(policy, user)
        new = to_chromadb_filter_v2(policy, user)

        assert original == new


class TestFilterBuilderBaseInterface:
    """Test the FilterBuilderBase interface and error handling."""

    def test_unsupported_operator_raises(self):
        """Test that unsupported operators raise UnsupportedConditionError."""
        from ragguard.exceptions import UnsupportedConditionError
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        builder = PineconeFilterBuilder()

        # Create a condition with an unusual combination
        # This tests the error handling path - literal == literal is not supported
        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(
                value_type=ValueType.LITERAL_STRING,
                value="literal",
                field_path=(),
            ),
            right=CompiledValue(
                value_type=ValueType.LITERAL_STRING,
                value="another_literal",
                field_path=(),
            ),
            original="'literal' == 'another_literal'",
        )

        with pytest.raises(UnsupportedConditionError):
            builder._build_from_condition(condition, {})

    def test_custom_backend_name(self):
        """Test that custom backend names are used in errors."""

        class CustomBuilder(DictFilterBuilder):
            backend_name = "my_custom_db"

        builder = CustomBuilder()
        assert builder.backend_name == "my_custom_db"


class TestDictFilterBuilderOperators:
    """Test DictFilterBuilder operator customization."""

    def test_default_operators(self):
        """Test default operator names."""
        builder = DictFilterBuilder()

        assert builder.op_eq == "$eq"
        assert builder.op_ne == "$ne"
        assert builder.op_in == "$in"
        assert builder.op_nin == "$nin"
        assert builder.op_and == "$and"
        assert builder.op_or == "$or"

    def test_custom_operators(self):
        """Test customizing operator names."""

        class MongoLikeBuilder(DictFilterBuilder):
            backend_name = "mongo"
            op_eq = "=="
            op_ne = "!="

        builder = MongoLikeBuilder()
        assert builder._filter_equals("field", "value") == {"field": {"==": "value"}}
        assert builder._filter_not_equals("field", "value") == {"field": {"!=": "value"}}


class TestOrAndExpressions:
    """Test OR/AND expression handling in the base class."""

    def test_or_expression(self):
        """Test native OR expression."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="or_test",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=[
                            "(document.status == 'published' OR document.public == true)"
                        ],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new
        assert "$or" in str(new)

    def test_and_expression(self):
        """Test native AND expression."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="and_test",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=[
                            "(document.status == 'published' AND document.reviewed == true)"
                        ],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        original = to_pinecone_filter(policy, user)
        new = to_pinecone_filter_v2(policy, user)

        assert original == new
        assert "$and" in str(new)


class TestComparisonOperators:
    """Test comparison operators in the base class."""

    def test_less_than(self):
        """Test < operator."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="priority_filter",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.priority < 5"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        new = to_pinecone_filter_v2(policy, user)
        assert new == {"priority": {"$lt": 5}}

    def test_greater_than_or_equal(self):
        """Test >= operator."""
        policy = Policy(
            version="1",
            rules=[
                Rule(
                    name="level_filter",
                    allow=AllowConditions(
                        everyone=True,
                        conditions=["document.level >= 3"],
                    ),
                )
            ],
        )
        user = {"id": "alice"}

        new = to_pinecone_filter_v2(policy, user)
        assert new == {"level": {"$gte": 3}}
