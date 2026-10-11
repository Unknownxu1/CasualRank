"""Exception types for CausalRank.

Phase 0/1 scope. Keep the hierarchy small: user/input errors are
``CausalRankValidationError``; everything else that the library refuses to do
falls back to ``CausalRankError``.
"""

from __future__ import annotations


class CausalRankError(Exception):
    """Base class for all CausalRank errors."""


class CausalRankValidationError(CausalRankError, ValueError):
    """Raised when user input (DataFrame schema, config) violates the contract.

    Subclasses ``ValueError`` so that ordinary data plumbing code can catch it
    as a value error without importing this package.
    """
