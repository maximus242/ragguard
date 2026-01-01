"""
Configuration classes for RAGGuard retrievers.

Provides a clean, type-safe way to configure retrievers with sensible defaults.
Supports configuration via environment variables for 12-factor compliance.

Environment Variables:
    RAGGUARD_ENABLE_VALIDATION: Enable input validation (default: true)
    RAGGUARD_ENABLE_RETRY: Enable retry logic (default: true)
    RAGGUARD_MAX_RETRIES: Maximum retry attempts (default: 3)
    RAGGUARD_RETRY_INITIAL_DELAY: Initial retry delay in seconds (default: 0.1)
    RAGGUARD_RETRY_MAX_DELAY: Maximum retry delay in seconds (default: 10.0)
    RAGGUARD_REQUEST_TIMEOUT: Request timeout in seconds (default: 30.0)
    RAGGUARD_CACHE_SIZE: Filter cache size (default: 1000)
    RAGGUARD_ENABLE_CACHE: Enable filter caching (default: true)
    RAGGUARD_ENABLE_AUDIT: Enable audit logging (default: false)
    RAGGUARD_LOG_LEVEL: Logging level (default: INFO)
    RAGGUARD_LOG_FORMAT: Log format - json or text (default: json)
    RAGGUARD_MAX_WORKERS: Max thread pool workers for async (default: 32)
"""

import os
from dataclasses import dataclass
from typing import Optional

from .retry import RetryConfig
from .validation import ValidationConfig


def _env_bool(key: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    value = os.environ.get(key, "").lower()
    if not value:
        return default
    return value in ("true", "1", "yes", "on")


def _env_int(key: str, default: int) -> int:
    """Parse integer from environment variable."""
    value = os.environ.get(key, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Parse float from environment variable."""
    value = os.environ.get(key, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(key, default)


@dataclass
class SecureRetrieverConfig:
    """
    Configuration for secure retrievers.

    Consolidates all retriever options into a single, reusable config object.
    Use this instead of passing many constructor arguments.

    Example:
        ```python
        from ragguard import SecureRetrieverConfig, QdrantSecureRetriever

        # Create reusable config
        config = SecureRetrieverConfig(
            enable_validation=True,
            filter_cache_size=5000,
            enable_retry=True
        )

        # Use with any retriever
        retriever1 = QdrantSecureRetriever(client, "docs", policy, config=config)
        retriever2 = ChromaDBSecureRetriever(collection, policy, config=config)
        ```

    For quick customization, use class methods:
        ```python
        # High-security mode (strict validation)
        config = SecureRetrieverConfig.strict()

        # Development mode (minimal overhead)
        config = SecureRetrieverConfig.development()

        # Production mode (balanced)
        config = SecureRetrieverConfig.production()
        ```
    """

    # Validation settings
    enable_validation: bool = True
    validation_config: Optional[ValidationConfig] = None

    # Retry settings
    enable_retry: bool = True
    retry_config: Optional[RetryConfig] = None

    # Filter cache settings
    enable_filter_cache: bool = True
    filter_cache_size: int = 1000

    # Audit settings
    enable_audit: bool = False

    def __post_init__(self):
        """Apply defaults for nested configs and validate options."""
        if self.validation_config is None:
            self.validation_config = ValidationConfig()
        if self.retry_config is None:
            self.retry_config = RetryConfig()

        self._validate_config()

    def _validate_config(self):
        """Validate configuration options for incompatibilities."""
        errors = []

        if self.filter_cache_size <= 0:
            errors.append(f"filter_cache_size must be positive, got {self.filter_cache_size}")

        if self.enable_retry and self.retry_config:
            if self.retry_config.max_retries < 0:
                errors.append(f"max_retries cannot be negative, got {self.retry_config.max_retries}")
            if self.retry_config.initial_delay <= 0:
                errors.append(f"initial_delay must be positive, got {self.retry_config.initial_delay}")
            if self.retry_config.max_delay <= 0:
                errors.append(f"max_delay must be positive, got {self.retry_config.max_delay}")
            if self.retry_config.initial_delay > self.retry_config.max_delay:
                errors.append(
                    f"initial_delay ({self.retry_config.initial_delay}) cannot exceed "
                    f"max_delay ({self.retry_config.max_delay})"
                )
            if self.retry_config.request_timeout <= 0:
                errors.append(f"request_timeout must be positive, got {self.retry_config.request_timeout}")
            if self.retry_config.total_timeout <= 0:
                errors.append(f"total_timeout must be positive, got {self.retry_config.total_timeout}")
            if self.retry_config.request_timeout > self.retry_config.total_timeout:
                errors.append(
                    f"request_timeout ({self.retry_config.request_timeout}) cannot exceed "
                    f"total_timeout ({self.retry_config.total_timeout})"
                )

        if self.enable_validation and self.validation_config:
            if self.validation_config.max_dict_size <= 0:
                errors.append(f"max_dict_size must be positive, got {self.validation_config.max_dict_size}")
            if self.validation_config.max_string_length <= 0:
                errors.append(f"max_string_length must be positive, got {self.validation_config.max_string_length}")
            if self.validation_config.max_nesting_depth <= 0:
                errors.append(f"max_nesting_depth must be positive, got {self.validation_config.max_nesting_depth}")
            if self.validation_config.max_array_length <= 0:
                errors.append(f"max_array_length must be positive, got {self.validation_config.max_array_length}")

        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    @classmethod
    def development(cls) -> "SecureRetrieverConfig":
        """
        Development configuration with minimal overhead.

        - Validation: enabled (catch bugs early)
        - Retry: disabled (fail fast for debugging)
        - Filter cache: small (quick iteration)
        """
        return cls(
            enable_validation=True,
            enable_retry=False,
            filter_cache_size=100,
            enable_audit=False
        )

    @classmethod
    def production(cls) -> "SecureRetrieverConfig":
        """
        Production configuration with balanced settings.

        - Validation: enabled
        - Retry: enabled with exponential backoff
        - Filter cache: standard size
        """
        return cls(
            enable_validation=True,
            validation_config=ValidationConfig(
                max_dict_size=100,
                max_string_length=10000,
                max_nesting_depth=10
            ),
            enable_retry=True,
            retry_config=RetryConfig(
                max_retries=3,
                initial_delay=0.1,
                max_delay=10.0,
                exponential_base=2,
                jitter=True
            ),
            filter_cache_size=1000,
            enable_audit=False
        )

    @classmethod
    def strict(cls) -> "SecureRetrieverConfig":
        """
        High-security configuration with strict validation.

        - Validation: strict limits
        - Retry: enabled with shorter timeouts
        - Filter cache: larger (minimize recomputation)
        - Audit: enabled
        """
        return cls(
            enable_validation=True,
            validation_config=ValidationConfig(
                max_dict_size=50,
                max_string_length=5000,
                max_nesting_depth=5,
                max_array_length=100
            ),
            enable_retry=True,
            retry_config=RetryConfig(
                max_retries=2,
                initial_delay=0.05,
                max_delay=5.0
            ),
            filter_cache_size=2000,
            enable_audit=True
        )

    @classmethod
    def minimal(cls) -> "SecureRetrieverConfig":
        """
        Minimal configuration for testing or simple use cases.

        - Validation: disabled
        - Retry: disabled
        - Filter cache: disabled
        """
        return cls(
            enable_validation=False,
            enable_retry=False,
            enable_filter_cache=False,
            enable_audit=False
        )

    @classmethod
    def from_env(cls) -> "SecureRetrieverConfig":
        """
        Create configuration from environment variables.

        12-factor compliant configuration that reads all settings from
        environment variables. Useful for containerized deployments.

        Environment Variables:
            RAGGUARD_ENABLE_VALIDATION: Enable validation (default: true)
            RAGGUARD_ENABLE_RETRY: Enable retry logic (default: true)
            RAGGUARD_MAX_RETRIES: Max retry attempts (default: 3)
            RAGGUARD_RETRY_INITIAL_DELAY: Initial delay in seconds (default: 0.1)
            RAGGUARD_RETRY_MAX_DELAY: Max delay in seconds (default: 10.0)
            RAGGUARD_REQUEST_TIMEOUT: Request timeout in seconds (default: 30.0)
            RAGGUARD_TOTAL_TIMEOUT: Total timeout in seconds (default: 120.0)
            RAGGUARD_CACHE_SIZE: Filter cache size (default: 1000)
            RAGGUARD_ENABLE_CACHE: Enable filter caching (default: true)
            RAGGUARD_ENABLE_AUDIT: Enable audit logging (default: false)

        Example:
            ```bash
            export RAGGUARD_ENABLE_VALIDATION=true
            export RAGGUARD_MAX_RETRIES=5
            export RAGGUARD_REQUEST_TIMEOUT=15.0
            ```

            ```python
            from ragguard import SecureRetrieverConfig

            # Reads from environment
            config = SecureRetrieverConfig.from_env()
            ```

        Returns:
            SecureRetrieverConfig: Configuration populated from environment
        """
        # Read retry settings from environment
        retry_config = RetryConfig(
            max_retries=_env_int("RAGGUARD_MAX_RETRIES", 3),
            initial_delay=_env_float("RAGGUARD_RETRY_INITIAL_DELAY", 0.1),
            max_delay=_env_float("RAGGUARD_RETRY_MAX_DELAY", 10.0),
            request_timeout=_env_float("RAGGUARD_REQUEST_TIMEOUT", 30.0),
            total_timeout=_env_float("RAGGUARD_TOTAL_TIMEOUT", 120.0),
            jitter=True
        )

        # Read validation settings from environment
        validation_config = ValidationConfig(
            max_dict_size=_env_int("RAGGUARD_MAX_DICT_SIZE", 100),
            max_string_length=_env_int("RAGGUARD_MAX_STRING_LENGTH", 10000),
            max_nesting_depth=_env_int("RAGGUARD_MAX_NESTING_DEPTH", 10),
            max_array_length=_env_int("RAGGUARD_MAX_ARRAY_LENGTH", 1000)
        )

        return cls(
            enable_validation=_env_bool("RAGGUARD_ENABLE_VALIDATION", True),
            validation_config=validation_config,
            enable_retry=_env_bool("RAGGUARD_ENABLE_RETRY", True),
            retry_config=retry_config,
            enable_filter_cache=_env_bool("RAGGUARD_ENABLE_CACHE", True),
            filter_cache_size=_env_int("RAGGUARD_CACHE_SIZE", 1000),
            enable_audit=_env_bool("RAGGUARD_ENABLE_AUDIT", False)
        )

    def with_validation(
        self,
        max_dict_size: Optional[int] = None,
        max_string_length: Optional[int] = None,
        max_nesting_depth: Optional[int] = None,
        max_array_length: Optional[int] = None
    ) -> "SecureRetrieverConfig":
        """
        Return a new config with updated validation settings.

        Example:
            config = SecureRetrieverConfig.production().with_validation(
                max_string_length=20000
            )
        """
        new_config = ValidationConfig(
            max_dict_size=max_dict_size if max_dict_size is not None else self.validation_config.max_dict_size,
            max_string_length=max_string_length if max_string_length is not None else self.validation_config.max_string_length,
            max_nesting_depth=max_nesting_depth if max_nesting_depth is not None else self.validation_config.max_nesting_depth,
            max_array_length=max_array_length if max_array_length is not None else self.validation_config.max_array_length
        )
        return SecureRetrieverConfig(
            enable_validation=True,
            validation_config=new_config,
            enable_retry=self.enable_retry,
            retry_config=self.retry_config,
            enable_filter_cache=self.enable_filter_cache,
            filter_cache_size=self.filter_cache_size,
            enable_audit=self.enable_audit
        )

    def with_retry(
        self,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None
    ) -> "SecureRetrieverConfig":
        """
        Return a new config with updated retry settings.

        Example:
            config = SecureRetrieverConfig.production().with_retry(
                max_retries=5
            )
        """
        new_config = RetryConfig(
            max_retries=max_retries if max_retries is not None else self.retry_config.max_retries,
            initial_delay=initial_delay if initial_delay is not None else self.retry_config.initial_delay,
            max_delay=max_delay if max_delay is not None else self.retry_config.max_delay,
            exponential_base=self.retry_config.exponential_base,
            jitter=self.retry_config.jitter
        )
        return SecureRetrieverConfig(
            enable_validation=self.enable_validation,
            validation_config=self.validation_config,
            enable_retry=True,
            retry_config=new_config,
            enable_filter_cache=self.enable_filter_cache,
            filter_cache_size=self.filter_cache_size,
            enable_audit=self.enable_audit
        )

    def with_cache(self, size: int) -> "SecureRetrieverConfig":
        """
        Return a new config with updated cache size.

        Example:
            config = SecureRetrieverConfig.production().with_cache(5000)
        """
        return SecureRetrieverConfig(
            enable_validation=self.enable_validation,
            validation_config=self.validation_config,
            enable_retry=self.enable_retry,
            retry_config=self.retry_config,
            enable_filter_cache=True,
            filter_cache_size=size,
            enable_audit=self.enable_audit
        )

    def with_audit(self, enabled: bool = True) -> "SecureRetrieverConfig":
        """
        Return a new config with audit logging enabled/disabled.

        Example:
            config = SecureRetrieverConfig.production().with_audit()
        """
        return SecureRetrieverConfig(
            enable_validation=self.enable_validation,
            validation_config=self.validation_config,
            enable_retry=self.enable_retry,
            retry_config=self.retry_config,
            enable_filter_cache=self.enable_filter_cache,
            filter_cache_size=self.filter_cache_size,
            enable_audit=enabled
        )
