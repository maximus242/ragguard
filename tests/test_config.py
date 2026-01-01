"""
Tests for SecureRetrieverConfig.

Tests configuration presets and fluent API for retriever configuration.
"""

import pytest

from ragguard.config import SecureRetrieverConfig
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig


class TestSecureRetrieverConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self):
        """Test default config has expected values."""
        config = SecureRetrieverConfig()

        assert config.enable_validation is True
        assert config.enable_retry is True
        assert config.enable_filter_cache is True
        assert config.filter_cache_size == 1000
        assert config.enable_audit is False

    def test_default_nested_configs_created(self):
        """Test nested configs are created with defaults."""
        config = SecureRetrieverConfig()

        assert config.validation_config is not None
        assert isinstance(config.validation_config, ValidationConfig)
        assert config.retry_config is not None
        assert isinstance(config.retry_config, RetryConfig)


class TestSecureRetrieverConfigPresets:
    """Test configuration presets."""

    def test_development_preset(self):
        """Test development configuration preset."""
        config = SecureRetrieverConfig.development()

        assert config.enable_validation is True
        assert config.enable_retry is False
        assert config.filter_cache_size == 100
        assert config.enable_audit is False

    def test_production_preset(self):
        """Test production configuration preset."""
        config = SecureRetrieverConfig.production()

        assert config.enable_validation is True
        assert config.enable_retry is True
        assert config.filter_cache_size == 1000
        assert config.enable_audit is False

        # Check nested config values
        assert config.validation_config.max_dict_size == 100
        assert config.validation_config.max_string_length == 10000
        assert config.retry_config.max_retries == 3
        assert config.retry_config.jitter is True

    def test_strict_preset(self):
        """Test strict (high-security) configuration preset."""
        config = SecureRetrieverConfig.strict()

        assert config.enable_validation is True
        assert config.enable_retry is True
        assert config.filter_cache_size == 2000
        assert config.enable_audit is True

        # Strict has lower limits
        assert config.validation_config.max_dict_size == 50
        assert config.validation_config.max_string_length == 5000
        assert config.validation_config.max_nesting_depth == 5
        assert config.retry_config.max_retries == 2

    def test_minimal_preset(self):
        """Test minimal configuration preset."""
        config = SecureRetrieverConfig.minimal()

        assert config.enable_validation is False
        assert config.enable_retry is False
        assert config.enable_filter_cache is False
        assert config.enable_audit is False


class TestSecureRetrieverConfigFluentAPI:
    """Test fluent API for configuration modification."""

    def test_with_validation(self):
        """Test with_validation returns modified config."""
        base = SecureRetrieverConfig.production()
        modified = base.with_validation(max_string_length=20000)

        # Original unchanged
        assert base.validation_config.max_string_length == 10000

        # Modified has new value
        assert modified.validation_config.max_string_length == 20000
        assert modified.enable_validation is True

        # Other values preserved
        assert modified.validation_config.max_dict_size == 100

    def test_with_validation_multiple_params(self):
        """Test with_validation with multiple parameters."""
        config = SecureRetrieverConfig().with_validation(
            max_dict_size=200,
            max_string_length=50000,
            max_nesting_depth=15,
            max_array_length=500
        )

        assert config.validation_config.max_dict_size == 200
        assert config.validation_config.max_string_length == 50000
        assert config.validation_config.max_nesting_depth == 15
        assert config.validation_config.max_array_length == 500

    def test_with_retry(self):
        """Test with_retry returns modified config."""
        base = SecureRetrieverConfig.production()
        modified = base.with_retry(max_retries=5)

        # Original unchanged
        assert base.retry_config.max_retries == 3

        # Modified has new value
        assert modified.retry_config.max_retries == 5
        assert modified.enable_retry is True

    def test_with_retry_multiple_params(self):
        """Test with_retry with multiple parameters."""
        config = SecureRetrieverConfig().with_retry(
            max_retries=10,
            initial_delay=0.5,
            max_delay=30.0
        )

        assert config.retry_config.max_retries == 10
        assert config.retry_config.initial_delay == 0.5
        assert config.retry_config.max_delay == 30.0

    def test_with_cache(self):
        """Test with_cache returns modified config."""
        base = SecureRetrieverConfig()
        modified = base.with_cache(5000)

        # Original unchanged
        assert base.filter_cache_size == 1000

        # Modified has new value
        assert modified.filter_cache_size == 5000
        assert modified.enable_filter_cache is True

    def test_with_audit_enabled(self):
        """Test with_audit enables audit logging."""
        base = SecureRetrieverConfig()
        modified = base.with_audit()

        # Original unchanged
        assert base.enable_audit is False

        # Modified has audit enabled
        assert modified.enable_audit is True

    def test_with_audit_disabled(self):
        """Test with_audit can disable audit logging."""
        base = SecureRetrieverConfig.strict()  # Has audit enabled
        modified = base.with_audit(enabled=False)

        # Original unchanged
        assert base.enable_audit is True

        # Modified has audit disabled
        assert modified.enable_audit is False

    def test_chained_modifications(self):
        """Test chaining multiple modifications."""
        config = (
            SecureRetrieverConfig.production()
            .with_validation(max_string_length=50000)
            .with_retry(max_retries=5)
            .with_cache(10000)
            .with_audit()
        )

        assert config.validation_config.max_string_length == 50000
        assert config.retry_config.max_retries == 5
        assert config.filter_cache_size == 10000
        assert config.enable_audit is True

        # Other production values preserved
        assert config.enable_validation is True
        assert config.enable_retry is True


class TestSecureRetrieverConfigImmutability:
    """Test that fluent API returns new instances."""

    def test_with_validation_returns_new_instance(self):
        """Test with_validation returns new config instance."""
        original = SecureRetrieverConfig()
        modified = original.with_validation(max_dict_size=999)

        assert original is not modified
        assert original.validation_config.max_dict_size != 999

    def test_with_retry_returns_new_instance(self):
        """Test with_retry returns new config instance."""
        original = SecureRetrieverConfig()
        modified = original.with_retry(max_retries=99)

        assert original is not modified
        assert original.retry_config.max_retries != 99

    def test_with_cache_returns_new_instance(self):
        """Test with_cache returns new config instance."""
        original = SecureRetrieverConfig()
        modified = original.with_cache(9999)

        assert original is not modified
        assert original.filter_cache_size != 9999

    def test_with_audit_returns_new_instance(self):
        """Test with_audit returns new config instance."""
        original = SecureRetrieverConfig()
        modified = original.with_audit()

        assert original is not modified
        assert original.enable_audit is False


class TestSecureRetrieverConfigImport:
    """Test config is properly exported from main package."""

    def test_import_from_ragguard(self):
        """Test SecureRetrieverConfig can be imported from ragguard."""
        from ragguard import SecureRetrieverConfig as ImportedConfig

        assert ImportedConfig is SecureRetrieverConfig

    def test_in_all_exports(self):
        """Test SecureRetrieverConfig is in __all__."""
        import ragguard

        assert "SecureRetrieverConfig" in ragguard.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
