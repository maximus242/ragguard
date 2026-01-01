"""
Condition compiler for RAGGuard policy engine.

Compiles string conditions to AST-like tokens at initialization time to avoid
runtime string parsing overhead.
"""

import logging
import math
import re
from typing import List, Union

from .models import (
    CompiledCondition,
    CompiledExpression,
    CompiledValue,
    ConditionOperator,
    ConditionType,
    LogicalOperator,
    ValueType,
)

logger = logging.getLogger(__name__)


def _unescape_string(s: str) -> str:
    """
    Unescape a string that may contain escape sequences.

    Handles common escape sequences:
    - \\\\ -> \\
    - \\' -> '
    - \\" -> "
    - \\n -> newline
    - \\t -> tab

    Args:
        s: String potentially containing escape sequences

    Returns:
        Unescaped string
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == '\\':
                result.append('\\')
                i += 2
            elif next_char == "'":
                result.append("'")
                i += 2
            elif next_char == '"':
                result.append('"')
                i += 2
            elif next_char == 'n':
                result.append('\n')
                i += 2
            elif next_char == 't':
                result.append('\t')
                i += 2
            elif next_char == 'r':
                result.append('\r')
                i += 2
            else:
                # Unknown escape - keep as-is
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


class ConditionCompiler:
    """
    Compiles string conditions to efficient AST-like representations.

    This is done once at PolicyEngine initialization time to avoid
    repeated string parsing during query evaluation.

    Supports both simple conditions and complex boolean expressions with OR/AND.
    """

    @staticmethod
    def compile_expression(condition: str) -> Union[CompiledCondition, CompiledExpression]:
        """
        Compile a condition string that may contain OR/AND logic.

        This is the main entry point that handles both:
        - Simple conditions: "user.dept == document.dept"
        - Complex expressions: "(user.role == 'admin' OR user.role == 'manager')"

        Args:
            condition: Condition string

        Returns:
            CompiledCondition or CompiledExpression depending on complexity

        Examples:
            >>> compile_expression("user.dept == document.dept")
            CompiledCondition(...)

            >>> compile_expression("(user.role == 'admin' OR user.role == 'manager')")
            CompiledExpression(OR, 2 children)
        """
        condition = condition.strip()
        original = condition

        # Check if condition contains OR/AND logic (outside of quotes)
        has_or_and = ConditionCompiler._contains_logical_operators(condition)

        if not has_or_and:
            # Simple condition - use existing logic
            return ConditionCompiler.compile_condition(condition)

        # Complex expression with OR/AND - parse into expression tree
        expr = ConditionCompiler._parse_expression(condition, original)

        # Validate total condition count (only for CompiledExpression)
        from ..models import PolicyLimits
        if isinstance(expr, CompiledExpression):
            condition_count = expr.count_conditions()
        else:
            condition_count = 1  # Single condition
        if condition_count > PolicyLimits.MAX_EXPRESSION_CONDITIONS:
            raise ValueError(
                f"Too many conditions in expression: {condition_count} > {PolicyLimits.MAX_EXPRESSION_CONDITIONS}\n"
                f"  {original}\n"
                f"  This could cause performance issues or DoS attacks.\n"
                f"  Try splitting into multiple rules."
            )

        return expr

    @staticmethod
    def _contains_logical_operators(condition: str) -> bool:
        """
        Check if condition contains OR/AND operators outside of quoted strings.

        Args:
            condition: Condition string

        Returns:
            True if contains OR/AND operators
        """
        # Remove quoted strings to avoid false positives
        cleaned = ConditionCompiler._remove_quoted_strings(condition)

        # Check for OR/AND as whole words (not part of other words)
        return bool(re.search(r'\bOR\b|\bAND\b', cleaned, re.IGNORECASE))

    @staticmethod
    def _remove_quoted_strings(condition: str) -> str:
        """
        Remove quoted strings from condition to avoid parsing quotes.

        Args:
            condition: Condition string

        Returns:
            Condition with quoted strings replaced by placeholders
        """
        # Replace single-quoted strings with placeholder
        cleaned = re.sub(r"'[^']*'", "''", condition)
        # Replace double-quoted strings with placeholder
        cleaned = re.sub(r'"[^"]*"', '""', cleaned)
        return cleaned

    @staticmethod
    def _parse_expression(condition: str, original: str, depth: int = 0) -> Union[CompiledCondition, CompiledExpression]:
        """
        Parse a complex boolean expression into an expression tree.

        Handles:
        - Parentheses for grouping
        - OR and AND operators
        - Nested expressions

        Args:
            condition: Condition string to parse
            original: Original condition for error messages
            depth: Current nesting depth (for limit checking)

        Returns:
            CompiledExpression tree

        Raises:
            ValueError: If expression syntax is invalid or exceeds complexity limits
        """
        from ..models import PolicyLimits

        condition = condition.strip()

        # Check depth limit to prevent stack overflow
        if depth >= PolicyLimits.MAX_EXPRESSION_DEPTH:
            raise ValueError(
                f"Expression nesting too deep: {depth} >= {PolicyLimits.MAX_EXPRESSION_DEPTH}\n"
                f"  {original}\n"
                f"  This could cause performance issues or stack overflow.\n"
                f"  Try simplifying your expression or splitting into multiple rules."
            )

        # Check for balanced parentheses
        if not ConditionCompiler._check_balanced_parens(condition):
            from ..errors import PolicyErrorFormatter
            error_msg = PolicyErrorFormatter.format_unbalanced_parens_error(original)
            raise ValueError(error_msg)

        # If wrapped in parens, unwrap
        if condition.startswith('(') and condition.endswith(')'):
            # Make sure these parens match
            if ConditionCompiler._find_matching_paren(condition, 0) == len(condition) - 1:
                condition = condition[1:-1].strip()

        # Find top-level OR/AND operators (not inside parentheses)
        # OR has lower precedence than AND, so look for OR first
        or_positions = ConditionCompiler._find_top_level_operator(condition, 'OR')

        if or_positions:
            # Split by OR
            children: List[Union[CompiledCondition, CompiledExpression]] = []
            parts = ConditionCompiler._split_by_positions(condition, or_positions)

            # Check branch count limit
            if len(parts) > PolicyLimits.MAX_EXPRESSION_BRANCHES:
                raise ValueError(
                    f"Too many OR branches: {len(parts)} > {PolicyLimits.MAX_EXPRESSION_BRANCHES}\n"
                    f"  {original}\n"
                    f"  This could cause performance issues or DoS attacks.\n"
                    f"  Try splitting into multiple rules or using 'in' operator for lists."
                )

            for part in parts:
                child = ConditionCompiler._parse_expression(part, original, depth + 1)
                children.append(child)

            return CompiledExpression(
                operator=LogicalOperator.OR,
                children=children,
                original=original
            )

        # No OR found, look for AND
        and_positions = ConditionCompiler._find_top_level_operator(condition, 'AND')

        if and_positions:
            # Split by AND
            children: List[Union[CompiledCondition, CompiledExpression]] = []
            parts = ConditionCompiler._split_by_positions(condition, and_positions)

            # Check branch count limit
            if len(parts) > PolicyLimits.MAX_EXPRESSION_BRANCHES:
                raise ValueError(
                    f"Too many AND branches: {len(parts)} > {PolicyLimits.MAX_EXPRESSION_BRANCHES}\n"
                    f"  {original}\n"
                    f"  This could cause performance issues or DoS attacks.\n"
                    f"  Try splitting into multiple rules."
                )

            for part in parts:
                child = ConditionCompiler._parse_expression(part, original, depth + 1)
                children.append(child)

            return CompiledExpression(
                operator=LogicalOperator.AND,
                children=children,
                original=original
            )

        # No OR/AND found - must be a simple condition
        # (This shouldn't happen if _contains_logical_operators was correct)
        return ConditionCompiler.compile_condition(condition)

    @staticmethod
    def _check_balanced_parens(condition: str) -> bool:
        """Check if parentheses are balanced."""
        depth = 0
        in_single_quote = False
        in_double_quote = False

        for char in condition:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '(' and not in_single_quote and not in_double_quote:
                depth += 1
            elif char == ')' and not in_single_quote and not in_double_quote:
                depth -= 1
                if depth < 0:
                    return False

        return depth == 0 and not in_single_quote and not in_double_quote

    @staticmethod
    def _find_matching_paren(condition: str, start: int) -> int:
        """Find the closing paren that matches the opening paren at start."""
        depth = 0
        in_single_quote = False
        in_double_quote = False

        for i in range(start, len(condition)):
            char = condition[i]

            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '(' and not in_single_quote and not in_double_quote:
                depth += 1
            elif char == ')' and not in_single_quote and not in_double_quote:
                depth -= 1
                if depth == 0:
                    return i

        return -1

    @staticmethod
    def _find_top_level_operator(condition: str, operator: str) -> list[int]:
        """
        Find positions of operator that are not inside parentheses or quotes.

        Args:
            condition: Condition string
            operator: Operator to find (e.g., 'OR', 'AND')

        Returns:
            List of character positions where operator appears at top level
        """
        positions = []
        depth = 0
        in_single_quote = False
        in_double_quote = False

        i = 0
        while i < len(condition):
            char = condition[i]

            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '(' and not in_single_quote and not in_double_quote:
                depth += 1
            elif char == ')' and not in_single_quote and not in_double_quote:
                depth -= 1
            elif depth == 0 and not in_single_quote and not in_double_quote:
                # Check if we're at the operator
                remaining = condition[i:]
                match = re.match(r'\b' + operator + r'\b', remaining, re.IGNORECASE)
                if match:
                    positions.append(i)
                    i += len(match.group(0)) - 1  # Skip past operator

            i += 1

        return positions

    @staticmethod
    def _split_by_positions(condition: str, positions: list[int]) -> list[str]:
        """
        Split condition string at given positions.

        Args:
            condition: Condition string
            positions: List of positions to split at (start of operator)

        Returns:
            List of condition parts
        """
        if not positions:
            return [condition]

        parts = []
        start = 0

        for pos in positions:
            # Extract part before operator
            part = condition[start:pos].strip()
            if part:
                parts.append(part)

            # Skip past operator (find next non-whitespace after operator word)
            # Operator is typically 2-3 chars ('OR', 'AND')
            i = pos
            while i < len(condition) and condition[i].isalpha():
                i += 1
            while i < len(condition) and condition[i].isspace():
                i += 1
            start = i

        # Add final part
        part = condition[start:].strip()
        if part:
            parts.append(part)

        return parts

    @staticmethod
    def compile_condition(condition: str) -> CompiledCondition:
        """
        Compile a condition string to a CompiledCondition.

        Args:
            condition: Condition string like "user.dept == document.dept"

        Returns:
            Compiled condition with pre-parsed operator and values

        Raises:
            ValueError: If condition syntax is invalid

        Examples:
            >>> compile_condition("user.dept == document.dept")
            CompiledCondition(EQUALS, user.dept, document.dept)

            >>> compile_condition("user.id in document.shared_with")
            CompiledCondition(IN, user.id, document.shared_with)
        """
        condition = condition.strip()
        original = condition

        # Check for common operator mistakes before parsing
        common_mistakes = {
            "===": "Use '==' instead of '===' (strict equality not needed in Python)",
            "!==": "Use '!=' instead of '!==' (strict inequality not needed in Python)",
            "<>": "Use '!=' instead of '<>' (SQL-style inequality not supported)",
            " = ": "Use '==' instead of '=' (assignment operator not supported in conditions)",
            " not_in ": "Use 'not in' (with space) instead of 'not_in'",
        }

        for mistake, suggestion in common_mistakes.items():
            if mistake in condition:
                from ..errors import PolicyErrorFormatter
                position = condition.index(mistake)
                error_msg = PolicyErrorFormatter.format_operator_error(mistake, condition, position)
                raise ValueError(error_msg)

        # Parse operator (order matters - check multi-char operators before single-char)
        if " not exists" in condition:
            operator = ConditionOperator.NOT_EXISTS
            left_str = condition.replace(" not exists", "").strip()
            right_str = None
        elif " exists" in condition:
            operator = ConditionOperator.EXISTS
            left_str = condition.replace(" exists", "").strip()
            right_str = None
        elif " not in " in condition:
            operator = ConditionOperator.NOT_IN
            left_str, right_str = condition.split(" not in ", 1)
        elif " in " in condition:
            operator = ConditionOperator.IN
            left_str, right_str = condition.split(" in ", 1)
        elif ">=" in condition:
            operator = ConditionOperator.GREATER_THAN_OR_EQUAL
            left_str, right_str = condition.split(">=", 1)
        elif "<=" in condition:
            operator = ConditionOperator.LESS_THAN_OR_EQUAL
            left_str, right_str = condition.split("<=", 1)
        elif ">" in condition:
            operator = ConditionOperator.GREATER_THAN
            left_str, right_str = condition.split(">", 1)
        elif "<" in condition:
            operator = ConditionOperator.LESS_THAN
            left_str, right_str = condition.split("<", 1)
        elif "==" in condition:
            operator = ConditionOperator.EQUALS
            left_str, right_str = condition.split("==", 1)
        elif "!=" in condition:
            operator = ConditionOperator.NOT_EQUALS
            left_str, right_str = condition.split("!=", 1)
        else:
            # No recognized operator found
            from ..errors import PolicyErrorFormatter
            error_msg = PolicyErrorFormatter.format_no_operator_error(condition)
            raise ValueError(error_msg)

        # Compile left and right values
        left = ConditionCompiler._compile_value(left_str.strip())
        right = ConditionCompiler._compile_value(right_str.strip()) if right_str else None

        # Determine condition type for IN/NOT IN operations
        condition_type = None
        if operator in (ConditionOperator.IN, ConditionOperator.NOT_IN):
            # Check if right side is a literal list (normal case)
            if right.value_type == ValueType.LITERAL_LIST:
                condition_type = ConditionType.FIELD_IN_LIST
            # Check if right side is a document field (array field case)
            elif right.value_type == ValueType.DOCUMENT_FIELD:
                condition_type = ConditionType.VALUE_IN_ARRAY_FIELD
            # Otherwise treat as field-in-list for backwards compatibility
            else:
                condition_type = ConditionType.FIELD_IN_LIST

        return CompiledCondition(
            operator=operator,
            left=left,
            right=right,
            original=original,
            condition_type=condition_type
        )

    @staticmethod
    def _compile_value(expr: str) -> CompiledValue:
        """
        Compile a value expression to a CompiledValue.

        Args:
            expr: Value expression like "user.department" or "'engineering'"

        Returns:
            Compiled value with type and pre-split path

        Examples:
            >>> _compile_value("user.department")
            CompiledValue(USER_FIELD, path=["department"])

            >>> _compile_value("'engineering'")
            CompiledValue(LITERAL_STRING, "engineering")

            >>> _compile_value("42")
            CompiledValue(LITERAL_NUMBER, 42)
        """
        expr = expr.strip()

        # Handle user context fields
        if expr.startswith("user."):
            field_path = expr[5:]  # Remove "user." prefix
            path_parts = tuple(field_path.split("."))
            return CompiledValue(
                value_type=ValueType.USER_FIELD,
                value=None,  # Value resolved at runtime
                field_path=path_parts
            )

        # Handle document context fields
        if expr.startswith("document."):
            field_path = expr[9:]  # Remove "document." prefix
            path_parts = tuple(field_path.split("."))
            return CompiledValue(
                value_type=ValueType.DOCUMENT_FIELD,
                value=None,  # Value resolved at runtime
                field_path=path_parts
            )

        # Handle literal strings (quoted)
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            # Strip quotes and unescape any escape sequences
            raw_value = expr[1:-1]
            unescaped_value = _unescape_string(raw_value)
            return CompiledValue(
                value_type=ValueType.LITERAL_STRING,
                value=unescaped_value,
                field_path=()
            )

        # Handle boolean literals
        if expr.lower() == "true":
            return CompiledValue(
                value_type=ValueType.LITERAL_BOOL,
                value=True,
                field_path=()
            )
        if expr.lower() == "false":
            return CompiledValue(
                value_type=ValueType.LITERAL_BOOL,
                value=False,
                field_path=()
            )

        # Handle None literal
        if expr.lower() == "none" or expr.lower() == "null":
            return CompiledValue(
                value_type=ValueType.LITERAL_NONE,
                value=None,
                field_path=()
            )

        # Handle literal numbers
        try:
            # Parse the number
            # Check for float indicators: decimal point or scientific notation (e/E)
            if "." in expr or "e" in expr.lower():
                num_value = float(expr)
            else:
                num_value = int(expr)

            # Validate magnitude to prevent DoS attacks via extreme values
            # Max safe integer in JavaScript is 2^53 - 1 = 9007199254740991
            # For compatibility, use similar limits
            MAX_SAFE_INT = 9007199254740991
            MIN_SAFE_INT = -9007199254740991

            if isinstance(num_value, int):
                if abs(num_value) > MAX_SAFE_INT:
                    raise ValueError(
                        f"Integer literal too large: {num_value}. "
                        f"Maximum supported value is ±{MAX_SAFE_INT}"
                    )
            elif isinstance(num_value, float):
                # Check for special float values that could cause undefined behavior
                if math.isnan(num_value):
                    raise ValueError(
                        "Float literal cannot be NaN (not a number). "
                        "Use explicit null checks instead."
                    )
                if math.isinf(num_value):
                    raise ValueError(
                        "Float literal cannot be infinity. "
                        "Use explicit comparisons instead."
                    )
                if abs(num_value) > 1e308:  # Near float max
                    raise ValueError(
                        f"Float literal too large: {num_value}. "
                        f"Maximum supported value is ±1e308"
                    )

            return CompiledValue(
                value_type=ValueType.LITERAL_NUMBER,
                value=num_value,
                field_path=()
            )
        except ValueError:
            pass

        # Handle list literals: ['a', 'b', 'c']
        if expr.startswith('['):
            # Check for missing closing bracket
            if not expr.endswith(']'):
                raise ValueError(
                    f"Malformed list literal: missing closing bracket ']'\n"
                    f"  {expr}\n"
                    f"  Expected ']' at end of list"
                )

            content = expr[1:-1].strip()

            if not content:
                # Empty list
                return CompiledValue(
                    value_type=ValueType.LITERAL_LIST,
                    value=[],
                    field_path=()
                )

            # Check for nested lists
            if '[' in content or ']' in content:
                raise ValueError(
                    f"Nested lists are not supported\n"
                    f"  Found in: {expr}\n"
                    f"  List elements must be simple values (strings, numbers, booleans)"
                )

            # Check for unclosed quotes using proper quote-aware parsing
            # Handles escaped quotes correctly (e.g., "He said \"hello\"")
            in_single_quote = False
            in_double_quote = False
            i = 0
            while i < len(content):
                char = content[i]

                # Handle escape sequences
                if char == '\\' and i + 1 < len(content):
                    # Skip the escaped character
                    i += 2
                    continue

                # Track quote state
                if char == "'" and not in_double_quote:
                    in_single_quote = not in_single_quote
                elif char == '"' and not in_single_quote:
                    in_double_quote = not in_double_quote

                i += 1

            # Check for unclosed quotes
            if in_single_quote:
                raise ValueError(
                    f"Malformed list literal: unclosed single quote\n"
                    f"  {expr}\n"
                    f"  Check that all strings are properly quoted"
                )
            if in_double_quote:
                raise ValueError(
                    f"Malformed list literal: unclosed double quote\n"
                    f"  {expr}\n"
                    f"  Check that all strings are properly quoted"
                )

            # Split by comma and parse each item
            items = []
            try:
                for item in content.split(','):
                    item = item.strip()
                    if item:
                        # Recursively compile each item
                        compiled_item = ConditionCompiler._compile_value(item)
                        # Extract the value from the compiled item
                        items.append(compiled_item.value)
            except Exception as e:
                raise ValueError(
                    f"Error parsing list literal: {e!s}\n"
                    f"  In expression: {expr}\n"
                    f"  Each element should be a quoted string, number, or boolean"
                )

            return CompiledValue(
                value_type=ValueType.LITERAL_LIST,
                value=items,
                field_path=()
            )

        # If nothing else matches, treat as literal string (unquoted)
        # This provides backward compatibility with unquoted literals
        return CompiledValue(
            value_type=ValueType.LITERAL_STRING,
            value=expr,
            field_path=()
        )
