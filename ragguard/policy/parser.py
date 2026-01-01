"""
Policy parser for loading and validating YAML policy files.
"""

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from ..exceptions import PolicyError, PolicyValidationError
from ..types import PolicyDict
from .models import Policy


class PolicyParser:
    """Parses and validates RAGGuard policy definitions."""

    @staticmethod
    def from_file(path: Union[str, Path], validate: bool = True) -> Policy:
        """
        Load a policy from a YAML file.

        Args:
            path: Path to the YAML policy file
            validate: If True, runs semantic validation (default: True)

        Returns:
            Validated Policy object

        Raises:
            PolicyError: If file cannot be read or parsed
            PolicyValidationError: If policy structure is invalid
        """
        path = Path(path)

        if not path.exists():
            raise PolicyError(f"Policy file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PolicyError(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise PolicyError(f"Failed to read policy file: {e}")

        if data is None:
            raise PolicyError("Policy file is empty")

        return PolicyParser.from_dict(data, validate=validate)

    @staticmethod
    def from_dict(data: PolicyDict, validate: bool = True) -> Policy:
        """
        Parse a policy from a dictionary.

        Args:
            data: Dictionary containing policy definition
            validate: If True, runs semantic validation (default: True)

        Returns:
            Validated Policy object

        Raises:
            PolicyValidationError: If policy structure is invalid
        """
        try:
            # Use Policy.from_dict which includes semantic validation
            return Policy.from_dict(data, validate=validate)
        except ValidationError as e:
            # Convert Pydantic validation errors to our custom exception
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(l) for l in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")

            error_msg = "Policy validation failed:\n  " + "\n  ".join(errors)
            raise PolicyValidationError(error_msg)
        except Exception as e:
            # Re-raise PolicyValidationError as-is
            if isinstance(e, PolicyValidationError):
                raise
            raise PolicyValidationError(f"Failed to parse policy: {e}")

    @staticmethod
    def from_yaml_string(yaml_string: str, validate: bool = True) -> Policy:
        """
        Parse a policy from a YAML string.

        Args:
            yaml_string: YAML-formatted policy string
            validate: If True, runs semantic validation (default: True)

        Returns:
            Validated Policy object

        Raises:
            PolicyError: If YAML cannot be parsed
            PolicyValidationError: If policy structure is invalid
        """
        try:
            data = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise PolicyError(f"Failed to parse YAML: {e}")

        if data is None:
            raise PolicyError("Policy YAML is empty")

        return PolicyParser.from_dict(data, validate=validate)


# Convenience function for common use case
def load_policy(path: Union[str, Path], validate: bool = True) -> Policy:
    """
    Load a policy from a YAML file.

    This is a convenience function that wraps PolicyParser.from_file.

    Args:
        path: Path to the YAML policy file
        validate: If True, runs semantic validation (default: True)

    Returns:
        Validated Policy object

    Raises:
        PolicyError: If file cannot be read or parsed
        PolicyValidationError: If policy structure or semantics are invalid
    """
    return PolicyParser.from_file(path, validate=validate)
