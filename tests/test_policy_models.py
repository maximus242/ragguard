"""
Complete tests for policy models to achieve 95%+ coverage.
"""

import pytest


def test_allow_conditions_is_empty():
    """Test AllowConditions.is_empty property."""
    from ragguard.policy.models import AllowConditions

    # Empty conditions
    empty = AllowConditions()
    assert empty.is_empty is True

    # Empty list of roles
    empty_roles = AllowConditions(roles=[])
    assert empty_roles.is_empty is True

    # Empty list of conditions
    empty_conditions = AllowConditions(conditions=[])
    assert empty_conditions.is_empty is True

    # Non-empty conditions
    with_everyone = AllowConditions(everyone=True)
    assert with_everyone.is_empty is False

    with_roles = AllowConditions(roles=["admin"])
    assert with_roles.is_empty is False

    with_conditions = AllowConditions(conditions=["user.id == document.owner"])
    assert with_conditions.is_empty is False


def test_rule_match_validation_dict_value():
    """Test Rule validates match conditions don't have dict values."""
    from pydantic import ValidationError

    from ragguard.policy.models import Rule

    # Should raise error for dict value
    with pytest.raises(ValidationError, match="Match condition values must be simple types"):
        Rule(
            name="test",
            match={"metadata": {"nested": "value"}},  # Dict value not allowed
            allow={"everyone": True}
        )


def test_rule_match_validation_list_value():
    """Test Rule allows list values in match conditions."""
    from ragguard.policy.models import Rule

    # Should allow list values (for "in" checks)
    rule = Rule(
        name="test",
        match={"status": ["active", "pending"]},
        allow={"everyone": True}
    )

    assert rule.match == {"status": ["active", "pending"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
