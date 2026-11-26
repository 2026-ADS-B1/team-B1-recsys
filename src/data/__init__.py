"""Data package for pipeline loading, validation, and cleaning."""

from typing import Any

__all__ = [
    "DataValidationError",
    "check_duplicate_ratings",
    "check_missing_values",
    "check_orphan_ids",
    "check_rating_range",
    "validate_raw_data",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.data import validation as _v
        return getattr(_v, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
