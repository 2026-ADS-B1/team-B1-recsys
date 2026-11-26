"""Data package for pipeline loading, validation, and cleaning."""

from typing import Any

__all__ = [
    # Validation
    "DataValidationError",
    "check_duplicate_ratings",
    "check_missing_values",
    "check_orphan_ids",
    "check_rating_range",
    "validate_raw_data",
    # Cleaning
    "DataConfigError",
    "DataSettings",
    "clean_and_save_data",
    "filter_movies_catalog",
    "filter_ratings_by_threshold",
    "load_data_settings",
    "standardize_dtypes_and_columns",
    "track_clean_data_with_dvc",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in {
            "DataValidationError",
            "check_duplicate_ratings",
            "check_missing_values",
            "check_orphan_ids",
            "check_rating_range",
            "validate_raw_data",
        }:
            from src.data import validation as _v
            return getattr(_v, name)
        else:
            from src.data import cleaning as _c
            return getattr(_c, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
