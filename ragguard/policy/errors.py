"""Enhanced error messages for RAGGuard policy validation and evaluation."""

from typing import Optional


class PolicyErrorFormatter:
    """Format policy errors with helpful context and suggestions."""

    # ANSI color codes (disabled if not in TTY)
    COLORS_ENABLED = True
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable_colors(cls) -> None:
        """Disable colored output."""
        cls.COLORS_ENABLED = False

    @classmethod
    def _color(cls, text: str, color_code: str) -> str:
        """Apply color if enabled."""
        if not cls.COLORS_ENABLED:
            return text
        return f"{color_code}{text}{cls.RESET}"

    @classmethod
    def format_validation_error(
        cls,
        message: str,
        condition: str,
        position: Optional[int] = None,
        suggestion: Optional[str] = None,
        examples: Optional[list[str]] = None
    ) -> str:
        """Format a validation error with context.

        Args:
            message: Main error message
            condition: The condition string that failed
            position: Character position where error occurred (optional)
            suggestion: Suggested fix (optional)
            examples: List of correct examples (optional)

        Returns:
            Formatted error message
        """
        lines = []

        # Header
        lines.append("")
        lines.append(cls._color("Policy Validation Error", cls.RED + cls.BOLD))
        lines.append("")

        # Message
        lines.append(cls._color(message, cls.RED))
        lines.append("")

        # Show the problematic condition
        lines.append(cls._color("In condition:", cls.BOLD))
        lines.append(f"  {condition}")

        # Point to the error location if position provided
        if position is not None:
            pointer = " " * (position + 2) + "^"
            lines.append(cls._color(pointer, cls.RED))

        lines.append("")

        # Suggestion
        if suggestion:
            lines.append(cls._color("Suggestion:", cls.YELLOW + cls.BOLD))
            lines.append(f"  {suggestion}")
            lines.append("")

        # Examples
        if examples:
            lines.append(cls._color("Examples:", cls.GREEN + cls.BOLD))
            for example in examples:
                lines.append(f"  • {example}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def format_operator_error(
        cls,
        invalid_operator: str,
        condition: str,
        position: int
    ) -> str:
        """Format an invalid operator error."""
        # Map common mistakes to suggestions
        suggestions = {
            "===": ("Use '==' for equality", ["user.role == 'admin'"]),
            "!==": ("Use '!=' for inequality", ["document.status != 'archived'"]),
            "<>": ("Use '!=' for inequality (SQL-style <> not supported)", ["document.status != 'draft'"]),
            "=": ("Use '==' for comparison (single '=' is assignment)", ["document.active == true"]),
            "not_in": ("Use 'not in' with a space", ["document.status not in ['draft', 'deleted']"]),
            "&&": ("Use 'AND' for logical and", ["(user.role == 'admin' AND document.public == true)"]),
            "||": ("Use 'OR' for logical or", ["(user.role == 'admin' OR user.role == 'manager')"]),
            "!": ("Use '!=' for inequality or 'not in' for exclusion", [
                "document.status != 'archived'",
                "document.tags not in ['draft', 'test']"
            ]),
        }

        suggestion, examples = suggestions.get(
            invalid_operator.strip(),
            (
                f"'{invalid_operator}' is not a valid operator",
                ["user.field == value", "document.field in ['a', 'b']"]
            )
        )

        all_examples = examples + [
            "",
            "Supported operators:",
            "  Equality: ==, !=",
            "  Comparison: >, <, >=, <=",
            "  Membership: in, not in",
            "  Existence: exists, not exists",
            "  Logical: OR, AND (with parentheses)"
        ]

        return cls.format_validation_error(
            message=f"Invalid operator '{invalid_operator}'",
            condition=condition,
            position=position,
            suggestion=suggestion,
            examples=all_examples
        )

    @classmethod
    def format_unbalanced_parens_error(cls, condition: str) -> str:
        """Format an unbalanced parentheses error."""
        # Count opening and closing parens
        open_count = condition.count('(')
        close_count = condition.count(')')

        if open_count > close_count:
            missing = open_count - close_count
            suggestion = f"Add {missing} closing parenthes{'is' if missing == 1 else 'es'} ')'"
        else:
            missing = close_count - open_count
            suggestion = f"Remove {missing} extra closing parenthes{'is' if missing == 1 else 'es'} ')' or add {missing} opening '('"

        return cls.format_validation_error(
            message="Unbalanced parentheses in expression",
            condition=condition,
            suggestion=suggestion,
            examples=[
                "(user.role == 'admin' OR user.role == 'manager')",
                "((user.dept == 'eng' OR user.dept == 'sales') AND document.public == true)"
            ]
        )

    @classmethod
    def format_complexity_error(
        cls,
        error_type: str,
        current: int,
        limit: int,
        condition: str
    ) -> str:
        """Format a complexity limit error."""
        messages = {
            "depth": f"Expression nesting too deep: {current} levels > {limit} max",
            "conditions": f"Too many conditions: {current} conditions > {limit} max",
            "branches": f"Too many OR/AND branches: {current} branches > {limit} max",
            "list_size": f"List too large: {current} elements > {limit} max",
        }

        suggestions = {
            "depth": "Simplify nested expressions or split into multiple rules",
            "conditions": "Split into multiple rules or simplify logic",
            "branches": "Group related conditions or split into separate rules",
            "list_size": "Use a smaller list or store values in a database field",
        }

        examples = {
            "depth": [
                "Instead of: ((((A OR B) AND C) OR D) AND E)",
                "Use: (A OR B OR D) AND (C AND E)"
            ],
            "conditions": [
                "Instead of: rule with 60 conditions",
                "Split into: 3 rules with 20 conditions each"
            ],
            "branches": [
                "Instead of: (A OR B OR C OR D OR E OR F OR G OR H OR I OR J OR K)",
                "Use: document.type in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']"
            ],
            "list_size": [
                "Instead of: document.id in [1000 items]",
                "Store in: document.category (indexed field)"
            ],
        }

        return cls.format_validation_error(
            message=messages.get(error_type, f"Complexity limit exceeded: {current} > {limit}"),
            condition=condition,
            suggestion=suggestions.get(error_type, "Simplify the expression"),
            examples=examples.get(error_type, [])
        )

    @classmethod
    def format_no_operator_error(cls, condition: str) -> str:
        """Format a missing operator error."""
        return cls.format_validation_error(
            message="No valid operator found",
            condition=condition,
            suggestion="Every condition must have an operator that compares two values",
            examples=[
                "user.department == document.department",
                "document.status == 'published'",
                "user.id in document.authorized_users",
                "document.level >= 5",
                "document.reviewed_at exists"
            ]
        )

    @classmethod
    def format_list_parsing_error(
        cls,
        condition: str,
        error_details: str
    ) -> str:
        """Format a list parsing error."""
        return cls.format_validation_error(
            message=f"Malformed list literal: {error_details}",
            condition=condition,
            suggestion="Lists must be enclosed in square brackets with comma-separated items",
            examples=[
                "document.status in ['active', 'pending']",
                "document.priority in [1, 2, 3]",
                "user.roles in ['admin', 'manager', 'owner']"
            ]
        )


class EnhancedPolicyValidationError(ValueError):
    """Enhanced policy validation error with formatted message."""

    def __init__(self, formatted_message: str):
        super().__init__(formatted_message)
        self.formatted_message = formatted_message

    def __str__(self) -> str:
        return self.formatted_message


class EnhancedPolicyEvaluationError(Exception):
    """Enhanced policy evaluation error with context."""

    def __init__(self, message: str, rule_name: Optional[str] = None, condition: Optional[str] = None):
        self.message = message
        self.rule_name = rule_name
        self.condition = condition

        # Format the error message
        lines = [
            "",
            PolicyErrorFormatter._color("Policy Evaluation Error", PolicyErrorFormatter.RED + PolicyErrorFormatter.BOLD),
            "",
            message
        ]

        if rule_name:
            lines.append("")
            lines.append(f"In rule: {PolicyErrorFormatter._color(rule_name, PolicyErrorFormatter.BOLD)}")

        if condition:
            lines.append("")
            lines.append("In condition:")
            lines.append(f"  {condition}")

        lines.append("")

        super().__init__("\n".join(lines))
