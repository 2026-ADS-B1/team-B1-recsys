"""Unit tests for data cleaning and threshold filtering (US1.3 - Task 2).

Covers:
- Reading data.min_ratings from params.yaml (separate from serving.min_ratings).
- Removing users and movies below the EDA threshold (k >= 10, configurable).
- Halting pipeline if validation checks fail before cleaning.
- Saving clean data and execution report with problems found and ratings removed.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from api.config import load_settings
from src.data.cleaning import (
    DataConfigError,
    DataSettings,
    clean_and_save_data,
    filter_movies_catalog,
    filter_ratings_by_threshold,
    load_data_settings,
)
from src.data.validation import DataValidationError


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    """Create a sample ratings DataFrame with known user and movie frequencies.

    Frequencies:
    - User 1: 3 ratings (movies 10, 20, 30)
    - User 2: 3 ratings (movies 10, 20, 40)
    - User 3: 2 ratings (movies 10, 20)
    - User 4: 1 rating  (movie 10)

    Movie frequencies:
    - Movie 10: 4 ratings (users 1, 2, 3, 4)
    - Movie 20: 3 ratings (users 1, 2, 3)
    - Movie 30: 1 rating  (user 1)
    - Movie 40: 1 rating  (user 2)
    Total: 9 ratings
    """
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 4],
        "movieId": [10, 20, 30, 10, 20, 40, 10, 20, 10],
        "rating": [4.0, 5.0, 3.0, 4.0, 3.5, 2.0, 5.0, 4.5, 3.0],
        "timestamp": [100, 101, 102, 103, 104, 105, 106, 107, 108],
    })


@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    """Create a matching movies metadata catalog."""
    return pd.DataFrame({
        "movieId": [10, 20, 30, 40],
        "title": ["Movie 10 (1995)", "Movie 20 (1998)", "Movie 30 (2000)", "Movie 40 (2002)"],
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi"],
    })


# ============================================================================
# 1. Configuration & Separation Tests (Checklist item 1)
# ============================================================================


def test_load_data_settings_reads_from_params_yaml():
    """Verify data.min_ratings is read from params.yaml with default 10."""
    settings = load_data_settings("params.yaml")
    assert isinstance(settings, DataSettings)
    assert settings.min_ratings == 10
    assert settings.raw_dir == "data/raw"
    assert settings.processed_dir == "data/processed"


def test_data_min_ratings_separate_from_serving_min_ratings():
    """Verify data.min_ratings (10) is separate from serving.min_ratings (50)."""
    data_settings = load_data_settings("params.yaml")
    serving_settings = load_settings("params.yaml")

    assert data_settings.min_ratings == 10
    assert serving_settings.min_ratings == 50
    assert data_settings.min_ratings != serving_settings.min_ratings


def test_load_data_settings_custom_file(tmp_path):
    """Verify custom params.yaml reading."""
    custom_yaml = tmp_path / "test_params.yaml"
    custom_yaml.write_text(
        "data:\n"
        "  min_ratings: 15\n"
        "  raw_dir: custom/raw\n"
        "  processed_dir: custom/clean\n",
        encoding="utf-8",
    )
    settings = load_data_settings(custom_yaml)
    assert settings.min_ratings == 15
    assert settings.raw_dir == "custom/raw"
    assert settings.processed_dir == "custom/clean"


def test_load_data_settings_errors(tmp_path):
    """Verify proper errors for missing or invalid data settings."""
    # Missing data section
    p1 = tmp_path / "p1.yaml"
    p1.write_text("serving:\n  min_ratings: 50\n", encoding="utf-8")
    with pytest.raises(DataConfigError, match="Missing 'data' section"):
        load_data_settings(p1)

    # Missing data.min_ratings
    p2 = tmp_path / "p2.yaml"
    p2.write_text("data:\n  raw_dir: data/raw\n", encoding="utf-8")
    with pytest.raises(DataConfigError, match="Missing 'data.min_ratings'"):
        load_data_settings(p2)

    # Negative min_ratings
    p3 = tmp_path / "p3.yaml"
    p3.write_text("data:\n  min_ratings: -5\n", encoding="utf-8")
    with pytest.raises(DataConfigError, match="must be 1 or greater"):
        load_data_settings(p3)

    # Non-integer min_ratings
    p4 = tmp_path / "p4.yaml"
    p4.write_text("data:\n  min_ratings: 'ten'\n", encoding="utf-8")
    with pytest.raises(DataConfigError, match="must be an integer"):
        load_data_settings(p4)


# ============================================================================
# 2. Threshold Filtering Tests (Checklist item 2, AC3)
# ============================================================================


def test_filter_ratings_by_threshold_basic(sample_ratings_df):
    """Test removing users and movies with fewer than k ratings (k=2)."""
    # With k=2:
    # Movies with < 2 ratings: movie 30 (1 rating), movie 40 (1 rating) -> removed
    # User 4 has only 1 rating (movie 10) -> removed
    clean_df, stats = filter_ratings_by_threshold(sample_ratings_df, min_ratings=2)

    # Remaining movies must only be 10 and 20
    assert set(clean_df["movieId"]) == {10, 20}
    # Remaining users must only be 1, 2, 3
    assert set(clean_df["userId"]) == {1, 2, 3}

    assert stats["initial_ratings"] == 9
    assert stats["clean_ratings"] == 6
    assert stats["ratings_removed"] == 3
    assert stats["movies_removed"] == 2
    assert stats["users_removed"] == 1


def test_filter_ratings_k_equals_1_no_op(sample_ratings_df):
    """When k <= 1, no ratings are removed."""
    clean_df, stats = filter_ratings_by_threshold(sample_ratings_df, min_ratings=1)
    assert len(clean_df) == len(sample_ratings_df)
    assert stats["ratings_removed"] == 0
    assert stats["pct_ratings_removed"] == 0.0


def test_filter_ratings_strict_threshold(sample_ratings_df):
    """When k is higher than any item has, result is empty."""
    clean_df, stats = filter_ratings_by_threshold(sample_ratings_df, min_ratings=10)
    assert len(clean_df) == 0
    assert stats["clean_ratings"] == 0
    assert stats["ratings_removed"] == 9
    assert stats["pct_ratings_removed"] == 100.0


def test_filter_movies_catalog(sample_movies_df, sample_ratings_df):
    """Test filtering movies catalog to match clean ratings."""
    clean_ratings, _ = filter_ratings_by_threshold(sample_ratings_df, min_ratings=2)
    clean_movies = filter_movies_catalog(sample_movies_df, clean_ratings)

    assert set(clean_movies["movieId"]) == {10, 20}
    assert len(clean_movies) == 2


# ============================================================================
# 3. End-to-End Cleaning & Report Generation (Checklist items 3 & 4, AC3, AC4)
# ============================================================================


def test_clean_and_save_data_success(sample_ratings_df, sample_movies_df, tmp_path):
    """Test end-to-end cleaning, file saving and execution report."""
    out_dir = tmp_path / "processed"
    rep_file = out_dir / "cleaning_report.json"

    report = clean_and_save_data(
        ratings_df=sample_ratings_df,
        movies_df=sample_movies_df,
        min_ratings=2,
        output_dir=out_dir,
        report_path=rep_file,
    )

    # 1. Output files exist (Parquet primary, CSV companion)
    assert (out_dir / "ratings.parquet").exists()
    assert (out_dir / "movies.parquet").exists()
    assert (out_dir / "ratings.csv").exists()
    assert (out_dir / "movies.csv").exists()
    assert rep_file.exists()

    # 2. Verify clean ratings on disk and EDA Section 10 types
    saved_ratings = pd.read_parquet(out_dir / "ratings.parquet")
    assert len(saved_ratings) == 6
    assert set(saved_ratings["movie_id"]) == {10, 20}
    assert "user_id" in saved_ratings.columns
    assert "movie_id" in saved_ratings.columns
    assert saved_ratings["user_id"].dtype == np.int32
    assert saved_ratings["movie_id"].dtype == np.int32
    assert saved_ratings["rating"].dtype == np.float32

    # 3. Verify execution report contents (AC4)
    assert report["status"] == "SUCCESS"
    assert report["threshold_k"] == 2
    assert report["validation"]["status"] == "PASSED"
    assert report["validation"]["problems_found"]["missing_values"] == 0
    assert report["filtering_summary"]["initial_ratings"] == 9
    assert report["filtering_summary"]["clean_ratings"] == 6
    assert report["filtering_summary"]["ratings_removed"] == 3
    assert report["filtering_summary"]["pct_ratings_removed"] == pytest.approx(33.33, abs=0.1)

    # 4. JSON file on disk matches report dict
    with open(rep_file, encoding="utf-8") as f:
        loaded_report = json.load(f)
    assert loaded_report["threshold_k"] == 2
    assert loaded_report["filtering_summary"]["clean_ratings"] == 6


def test_clean_and_save_data_halts_on_validation_failure(sample_ratings_df, sample_movies_df, tmp_path):
    """Test AC1 & AC2: Bad data stops the pipeline before saving clean data."""
    bad_ratings = sample_ratings_df.copy()
    bad_ratings.loc[0, "rating"] = 9.0  # Invalid rating > 5.0

    out_dir = tmp_path / "processed"

    with pytest.raises(DataValidationError) as exc_info:
        clean_and_save_data(
            ratings_df=bad_ratings,
            movies_df=sample_movies_df,
            min_ratings=2,
            output_dir=out_dir,
        )

    # Pipeline stopped: clean data files should not have been created
    assert not (out_dir / "ratings.csv").exists()
    assert "outside [0.5, 5.0]" in str(exc_info.value)
