"""
Tests for retriever API compatibility utilities.
"""

import warnings

import pytest

from ragguard.retrievers.compat import (
    ParameterAlias,
    deprecate_param,
    standardize_kwargs,
)


class TestDeprecateParam:
    """Test the deprecate_param decorator."""

    def test_old_param_triggers_warning(self):
        """Test that using old parameter name triggers warning."""

        @deprecate_param("old_name", "new_name")
        def func(new_name: str):
            return new_name

        with pytest.warns(DeprecationWarning, match="old_name.*deprecated"):
            result = func(old_name="value")

        assert result == "value"

    def test_new_param_no_warning(self):
        """Test that using new parameter name doesn't trigger warning."""

        @deprecate_param("old_name", "new_name")
        def func(new_name: str):
            return new_name

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = func(new_name="value")

        assert result == "value"

    def test_both_params_uses_new(self):
        """Test that when both are specified, new takes precedence."""

        @deprecate_param("old_name", "new_name")
        def func(new_name: str):
            return new_name

        with pytest.warns(DeprecationWarning):
            result = func(old_name="old", new_name="new")

        assert result == "new"

    def test_preserves_other_kwargs(self):
        """Test that other kwargs are preserved."""

        @deprecate_param("old_name", "new_name")
        def func(new_name: str, other: str):
            return f"{new_name}-{other}"

        with pytest.warns(DeprecationWarning):
            result = func(old_name="value", other="extra")

        assert result == "value-extra"


class TestStandardizeKwargs:
    """Test the standardize_kwargs function."""

    def test_maps_deprecated_to_standard(self):
        """Test mapping deprecated names to standard names."""
        kwargs = {"collection": "my_collection"}
        mappings = {"collection": "collection_name"}

        with pytest.warns(DeprecationWarning):
            result = standardize_kwargs(kwargs, mappings)

        assert result == {"collection_name": "my_collection"}
        assert "collection" not in result

    def test_preserves_standard_names(self):
        """Test that standard names are preserved."""
        kwargs = {"collection_name": "my_collection"}
        mappings = {"collection": "collection_name"}

        # No warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = standardize_kwargs(kwargs, mappings)

        assert result == {"collection_name": "my_collection"}

    def test_both_names_uses_standard(self):
        """Test that when both are present, standard takes precedence."""
        kwargs = {
            "collection": "old_value",
            "collection_name": "new_value",
        }
        mappings = {"collection": "collection_name"}

        with pytest.warns(DeprecationWarning):
            result = standardize_kwargs(kwargs, mappings)

        assert result == {"collection_name": "new_value"}

    def test_preserves_unrelated_kwargs(self):
        """Test that unrelated kwargs are preserved."""
        kwargs = {
            "collection": "my_collection",
            "policy": "my_policy",
            "other": "value",
        }
        mappings = {"collection": "collection_name"}

        with pytest.warns(DeprecationWarning):
            result = standardize_kwargs(kwargs, mappings)

        assert result == {
            "collection_name": "my_collection",
            "policy": "my_policy",
            "other": "value",
        }

    def test_no_warning_when_disabled(self):
        """Test that warnings can be disabled."""
        kwargs = {"collection": "my_collection"}
        mappings = {"collection": "collection_name"}

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = standardize_kwargs(kwargs, mappings, warn=False)

        assert result == {"collection_name": "my_collection"}

    def test_multiple_mappings(self):
        """Test multiple parameter mappings."""
        kwargs = {
            "collection": "my_collection",
            "embedding_column": "vec",
        }
        mappings = {
            "collection": "collection_name",
            "embedding_column": "vector_field",
        }

        with pytest.warns(DeprecationWarning):
            result = standardize_kwargs(kwargs, mappings)

        assert result == {
            "collection_name": "my_collection",
            "vector_field": "vec",
        }


class TestParameterAlias:
    """Test the ParameterAlias descriptor."""

    def test_get_triggers_warning(self):
        """Test that getting aliased attribute triggers warning."""

        class TestClass:
            collection = ParameterAlias("collection_name")

            def __init__(self):
                self.collection_name = "my_collection"

        obj = TestClass()

        with pytest.warns(DeprecationWarning, match="collection.*deprecated"):
            value = obj.collection

        assert value == "my_collection"

    def test_set_triggers_warning(self):
        """Test that setting aliased attribute triggers warning."""

        class TestClass:
            collection = ParameterAlias("collection_name")

            def __init__(self):
                self.collection_name = "initial"

        obj = TestClass()

        with pytest.warns(DeprecationWarning, match="collection.*deprecated"):
            obj.collection = "new_value"

        assert obj.collection_name == "new_value"

    def test_standard_name_no_warning(self):
        """Test that using standard name doesn't trigger warning."""

        class TestClass:
            collection = ParameterAlias("collection_name")

            def __init__(self):
                self.collection_name = "my_collection"

        obj = TestClass()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = obj.collection_name

        assert value == "my_collection"
