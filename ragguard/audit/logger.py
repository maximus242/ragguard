"""
Audit logging for RAGGuard queries.

Logs all permission-aware searches for compliance and debugging.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logs all permission-aware queries for audit and compliance.

    Supports multiple output modes:
    - stdout: Print to console
    - file:path: Append to JSONL file
    - callback:func: Call custom function with log entry

    Warning:
        For compliance-critical applications (HIPAA, SOC2, GDPR), set
        raise_on_failure=True to ensure audit failures are not silently ignored.
        Monitor failure_count property to detect issues.
    """

    def __init__(
        self,
        output: str = "stdout",
        raise_on_failure: bool = False,
        on_failure: Optional[Any] = None
    ):
        """
        Initialize the audit logger.

        Args:
            output: Output mode
                - "stdout": Print to console (default)
                - "file:/path/to/file.jsonl": Append to file
                - Custom callable: Will be called with each log entry
            raise_on_failure: If True, raise exceptions on audit failures
                             instead of silently ignoring them. Recommended
                             for compliance-critical applications.
            on_failure: Optional callback(error, entry) called when audit fails.
                       Use for alerting or metrics collection.

        Warning:
            Default raise_on_failure=False means audit failures are logged but
            not raised. For compliance requirements, set raise_on_failure=True.
        """
        if callable(output):
            self.output = "callback"
            self.callback = output
        else:
            self.output = output
            self.callback = None

        self.raise_on_failure = raise_on_failure
        self.on_failure = on_failure
        self._failure_count = 0
        self._success_count = 0

        # Warn users about silent failure mode for compliance awareness
        if not raise_on_failure:
            logger.info(
                "AuditLogger initialized with raise_on_failure=False. "
                "Audit failures will be logged but not raised. "
                "For compliance-critical applications (HIPAA, SOC2, GDPR), "
                "consider setting raise_on_failure=True."
            )

    @property
    def failure_count(self) -> int:
        """Number of audit log failures since creation."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Number of successful audit logs since creation."""
        return self._success_count

    def _handle_failure(self, error: Exception, entry: dict[str, Any]) -> None:
        """Handle an audit failure."""
        self._failure_count += 1

        if self.on_failure:
            try:
                self.on_failure(error, entry)
            except Exception as callback_err:
                # Log the callback error but don't fail - callbacks shouldn't break auditing
                logger.warning(
                    "Audit on_failure callback raised an exception",
                    extra={"extra_fields": {"callback_error": str(callback_err)}}
                )

        if self.raise_on_failure:
            raise RuntimeError(f"Audit logging failed: {error}") from error

    def log(
        self,
        user: dict[str, Any],
        query: str,
        results_count: int,
        filter_applied: Any,
        additional_info: Optional[dict[str, Any]] = None
    ):
        """
        Log a query.

        Args:
            user: User context
            query: Query string or "[vector]" for vector queries
            results_count: Number of results returned
            filter_applied: The filter that was applied
            additional_info: Optional additional information to log
        """
        entry = {
            # Use timezone-aware datetime for proper UTC timestamps
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user.get("id"),
            "user_roles": user.get("roles", []),
            "query": query,
            "results_returned": results_count,
            "filter": str(filter_applied),
        }

        # Add any additional info
        if additional_info:
            entry.update(additional_info)

        self._write(entry)

    def _write(self, entry: dict[str, Any]):
        """Write a log entry to the configured output."""
        try:
            if self.output == "stdout":
                print(json.dumps(entry))

            elif self.output == "callback":
                if self.callback:
                    self.callback(entry)

            elif self.output.startswith("file:"):
                path_str = self.output[5:]
                # resolve() follows symlinks, so symlink attacks are prevented
                path = Path(path_str).resolve()

                # Protected system directories - deny writes to prevent security issues
                # Note: resolve() above ensures symlinks can't bypass this check
                dangerous_roots = (
                    '/',        # Root directory
                    '/etc',     # System configuration
                    '/usr',     # System programs
                    '/bin',     # Essential binaries
                    '/sbin',    # System binaries
                    '/var',     # Variable data (logs, mail, etc)
                    '/sys',     # Kernel/system info
                    '/proc',    # Process info
                    '/dev',     # Device files
                    '/boot',    # Boot files
                    '/lib',     # Essential libraries
                    '/lib64',   # 64-bit libraries
                    '/root',    # Root user home
                    '/opt',     # Optional software (may contain system apps)
                )
                path_str_resolved = str(path)
                if any(path_str_resolved.startswith(root + '/') or path_str_resolved == root for root in dangerous_roots):
                    raise ValueError(
                        f"Audit log path '{path}' is in a protected system directory. "
                        f"Please use a path in your application directory."
                    )

                path.parent.mkdir(parents=True, exist_ok=True)

                with open(path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            else:
                print(json.dumps(entry))

            self._success_count += 1

        except Exception as e:
            logger.error(
                "Audit logging failed",
                extra={"extra_fields": {"error": str(e), "user_id": entry.get("user_id")}},
                exc_info=True
            )
            self._handle_failure(e, entry)


# Null logger that does nothing (useful for disabling logging)
class NullAuditLogger(AuditLogger):
    """Audit logger that does nothing."""

    def __init__(self):
        super().__init__()

    def log(self, *args, **kwargs):
        """Do nothing."""
        pass

    def _write(self, entry: dict[str, Any]):
        """Do nothing."""
        pass
