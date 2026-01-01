"""Tests for ragguard.constants module."""

from ragguard.constants import (
    ALL_BACKENDS,
    ALL_OPERATORS,
    BACKEND_CHROMADB,
    # Backends
    BACKEND_QDRANT,
    BINARY_OPERATORS,
    # Cache defaults
    DEFAULT_CACHE_SIZE,
    # Sentinel values
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    DOCUMENT_PREFIX,
    DOCUMENT_PREFIX_LEN,
    # Security limits
    MAX_BACKEND_NAME_LENGTH,
    MAX_CONDITION_STRING_LENGTH,
    MAX_EXPRESSION_DEPTH,
    NATIVE_FILTER_BACKENDS,
    # Operators
    OP_EQUALS,
    OP_IN,
    OP_NOT_EQUALS,
    OP_NOT_IN,
    POST_FILTER_BACKENDS,
    SKIP_RULE,
    UNARY_OPERATORS,
    # Field prefixes
    USER_PREFIX,
    USER_PREFIX_LEN,
    _SkipRuleSentinel,
)


class TestFieldPrefixes:
    """Tests for field prefix constants."""

    def test_user_prefix(self):
        assert USER_PREFIX == "user."
        assert USER_PREFIX_LEN == 5
        assert USER_PREFIX_LEN == len(USER_PREFIX)

    def test_document_prefix(self):
        assert DOCUMENT_PREFIX == "document."
        assert DOCUMENT_PREFIX_LEN == 9
        assert DOCUMENT_PREFIX_LEN == len(DOCUMENT_PREFIX)


class TestSecurityLimits:
    """Tests for security limit constants."""

    def test_limits_are_reasonable(self):
        assert MAX_BACKEND_NAME_LENGTH > 10
        assert MAX_BACKEND_NAME_LENGTH <= 1000
        assert MAX_CONDITION_STRING_LENGTH > 100
        assert MAX_EXPRESSION_DEPTH > 5


class TestCacheDefaults:
    """Tests for cache default constants."""

    def test_cache_size(self):
        assert DEFAULT_CACHE_SIZE > 0
        assert DEFAULT_CACHE_SIZE <= 10000


class TestSentinelValues:
    """Tests for deny-all and skip-rule sentinel constants."""

    def test_sentinel_values_distinct(self):
        # Sentinels should be unlikely to collide with real field names
        assert "__" in DENY_ALL_FIELD
        assert "__" in DENY_ALL_VALUE
        assert DENY_ALL_FIELD != DENY_ALL_VALUE

    def test_skip_rule_is_singleton(self):
        # SKIP_RULE should be a singleton
        assert SKIP_RULE is _SkipRuleSentinel()
        assert _SkipRuleSentinel() is _SkipRuleSentinel()

    def test_skip_rule_is_distinct_from_none(self):
        # SKIP_RULE should be distinguishable from None
        assert SKIP_RULE is not None
        assert SKIP_RULE != None  # noqa: E711 - explicit None comparison for clarity

    def test_skip_rule_repr(self):
        assert repr(SKIP_RULE) == "SKIP_RULE"

    def test_skip_rule_identity(self):
        # Can use `is` for identity comparison
        assert SKIP_RULE is SKIP_RULE
        result = SKIP_RULE
        assert result is SKIP_RULE


class TestOperators:
    """Tests for operator constants."""

    def test_equality_operators(self):
        assert OP_EQUALS == "=="
        assert OP_NOT_EQUALS == "!="

    def test_membership_operators(self):
        assert " in " in OP_IN
        assert " not in " in OP_NOT_IN

    def test_all_operators_contains_common_ops(self):
        assert OP_EQUALS in ALL_OPERATORS
        assert OP_NOT_EQUALS in ALL_OPERATORS
        assert OP_IN in ALL_OPERATORS

    def test_binary_vs_unary(self):
        # Binary operators compare two values
        assert OP_EQUALS in BINARY_OPERATORS
        assert OP_IN in BINARY_OPERATORS
        # Unary operators check existence
        assert len(UNARY_OPERATORS) > 0
        for op in UNARY_OPERATORS:
            assert op not in BINARY_OPERATORS


class TestBackends:
    """Tests for backend name constants."""

    def test_backend_names(self):
        assert BACKEND_QDRANT == "qdrant"
        assert BACKEND_CHROMADB == "chromadb"

    def test_all_backends_not_empty(self):
        assert len(ALL_BACKENDS) >= 10  # We support at least 10 backends

    def test_native_vs_post_filter(self):
        # FAISS is post-filter only
        assert "faiss" in POST_FILTER_BACKENDS
        assert "faiss" not in NATIVE_FILTER_BACKENDS
        # Qdrant is native
        assert "qdrant" in NATIVE_FILTER_BACKENDS
        assert "qdrant" not in POST_FILTER_BACKENDS

    def test_all_backends_covered(self):
        # Every backend should be in either native or post-filter
        covered = set(NATIVE_FILTER_BACKENDS) | set(POST_FILTER_BACKENDS)
        assert set(ALL_BACKENDS) == covered
