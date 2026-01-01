"""
Condition compilation for RAGGuard policy engine.

Pre-compiles string conditions to AST-like tokens at initialization time to avoid
runtime string parsing overhead. This provides significant performance improvements
for condition evaluation.

This module re-exports all public APIs for backwards compatibility with code that
imports from ragguard.policy.compiler.
"""

# Re-export all models
# Re-export the compiler
from .condition_compiler import ConditionCompiler

# Re-export the evaluator
from .evaluator import CompiledConditionEvaluator
from .models import (
    CompiledCondition,
    CompiledExpression,
    CompiledValue,
    ConditionOperator,
    ConditionType,
    LogicalOperator,
    ValueType,
)

# Define public API
__all__ = [
    # Enums
    "ConditionOperator",
    "ValueType",
    "ConditionType",
    "LogicalOperator",
    # Data classes
    "CompiledValue",
    "CompiledCondition",
    "CompiledExpression",
    # Compiler
    "ConditionCompiler",
    # Evaluator
    "CompiledConditionEvaluator",
]
