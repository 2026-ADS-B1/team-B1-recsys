"""Unit tests for basic data validation checks (US1.3 - Task 1).

Covers:
- Missing values in each column of users, movies, and ratings.
- Ratings outside [min_rating, max_rating] (e.g., < 0.5 or > 5.0).
- Repeated user-movie ratings.
- User and movie IDs that don't exist in the other tables (orphan IDs).
- Halting pipeline with clear DataValidationError exception message.
- Small examples with known synthetic problems.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    check_duplicate_ratings,
    check_missing_values,
    check_orphan_ids,
    check_rating_range,
    validate_raw_data,
)


@pytest.fixture
def clean_ratings_df() -> pd.DataFrame:
    """Create a minimal valid ratings DataFrame."""
    return pd.DataFrame({
        "userId": [1, 1, 2, 2, 3],
        "movieId": [10, 20, 10, 30, 20],
        "rating": [4.0, 5.0, 3.5, 2.0, 4.5],
        "timestamp": [1000, 1001, 1002, 1003, 1004],
    })


@pytest.fixture
def clean_movies_df() -> pd.DataFrame:
    """Create a minimal valid movies catalog DataFrame."""
    return pd.DataFrame({
        "movieId": [10, 20, 30],
        "title": ["Movie A (1995)", "Movie B (1998)", "Movie C (2001)"],
        "genres": ["Action|Adventure", "Comedy", "Drama"],
    })


@pytest.fixture
def clean_users_df() -> pd.DataFrame:
    """Create a minimal valid users metadata DataFrame."""
    return pd.DataFrame({
        "userId": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "age": [25, 30, 45],
        "occupation": ["engineer", "artist", "programmer"],
    })


# ============================================================================
# 1. Happy Path & Report Structure Tests
# ============================================================================


def test_validate_clean_data_passes(clean_ratings_df, clean_movies_df):
    """Test AC1: Valid clean data passes all checks and returns a summary report."""
    report = validate_raw_data(clean_ratings_df, clean_movies_df)

    assert report["status"] == "PASSED"
    assert report["total_ratings"] == 5
    assert report["total_movies"] == 3
    assert report["total_users"] == 3
    assert report["rating_range"]["min_observed"] == 2.0
    assert report["rating_range"]["max_observed"] == 5.0
    assert "missing_values" in report["checks_passed"]
    assert "rating_range" in report["checks_passed"]
    assert "duplicate_ratings" in report["checks_passed"]
    assert "orphan_ids" in report["checks_passed"]


def test_validate_clean_data_with_users_passes(clean_ratings_df, clean_movies_df, clean_users_df):
    """Test validation with users metadata table included."""
    report = validate_raw_data(clean_ratings_df, clean_movies_df, users_df=clean_users_df)

    assert report["status"] == "PASSED"
    assert report["total_users"] == 3


def test_supports_snake_case_column_names(clean_movies_df):
    """Test that check handles snake_case column names (user_id, movie_id)."""
    snake_ratings = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [10, 20],
        "rating": [4.0, 5.0],
        "timestamp": [1000, 1001],
    })
    snake_movies = clean_movies_df.rename(columns={"movieId": "movie_id"})

    report = validate_raw_data(snake_ratings, snake_movies)
    assert report["status"] == "PASSED"


# ============================================================================
# 2. Missing Values Tests (AC2)
# ============================================================================


def test_missing_values_in_ratings_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Missing value in ratings halts pipeline with clear message."""
    bad_ratings = clean_ratings_df.copy()
    bad_ratings.loc[1, "rating"] = np.nan

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df)

    err = str(exc_info.value)
    assert "Data pipeline halted!" in err
    assert "Table 'ratings'" in err
    assert "column 'rating'" in err
    assert "missing value" in err


def test_missing_values_in_movies_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Missing value in movies table halts pipeline."""
    bad_movies = clean_movies_df.copy()
    bad_movies.loc[0, "title"] = np.nan

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(clean_ratings_df, bad_movies)

    err = str(exc_info.value)
    assert "Data pipeline halted!" in err
    assert "Table 'movies'" in err
    assert "column 'title'" in err


def test_missing_values_in_users_fails(clean_ratings_df, clean_movies_df, clean_users_df):
    """Test AC2: Missing value in users table halts pipeline when users provided."""
    bad_users = clean_users_df.copy()
    bad_users.loc[0, "occupation"] = np.nan

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(clean_ratings_df, clean_movies_df, users_df=bad_users)

    err = str(exc_info.value)
    assert "Data pipeline halted!" in err
    assert "Table 'users'" in err
    assert "column 'occupation'" in err


# ============================================================================
# 3. Rating Range Tests (AC2)
# ============================================================================


def test_rating_below_minimum_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Rating below allowed minimum (e.g. 0.0 or negative) halts pipeline."""
    bad_ratings = clean_ratings_df.copy()
    bad_ratings.loc[0, "rating"] = 0.0

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df)

    err = str(exc_info.value)
    assert "Table 'ratings'" in err
    assert "outside [0.5, 5.0]" in err


def test_rating_above_maximum_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Rating above allowed maximum (e.g. 6.0) halts pipeline."""
    bad_ratings = clean_ratings_df.copy()
    bad_ratings.loc[0, "rating"] = 6.0

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df)

    err = str(exc_info.value)
    assert "Table 'ratings'" in err
    assert "outside [0.5, 5.0]" in err
    assert "Max observed: 6.0" in err


def test_custom_rating_range_bounds(clean_ratings_df, clean_movies_df):
    """Test configurable bounds, e.g. strict [1.0, 5.0]."""
    test_ratings = clean_ratings_df.copy()
    test_ratings.loc[0, "rating"] = 0.5  # Allowed under 0.5-5.0, but forbidden under 1.0-5.0

    # With default 0.5-5.0, it passes:
    report = validate_raw_data(test_ratings, clean_movies_df, min_rating=0.5, max_rating=5.0)
    assert report["status"] == "PASSED"

    # With strict 1.0-5.0, it halts:
    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(test_ratings, clean_movies_df, min_rating=1.0, max_rating=5.0)

    assert "outside [1.0, 5.0]" in str(exc_info.value)


# ============================================================================
# 4. Repeated User-Movie Ratings Tests (AC2)
# ============================================================================


def test_duplicate_user_movie_ratings_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Repeated rating for the same (user, movie) pair halts pipeline."""
    bad_ratings = clean_ratings_df.copy()
    # Add duplicate of row 0: user 1, movie 10
    duplicate_row = pd.DataFrame({
        "userId": [1],
        "movieId": [10],
        "rating": [3.0],
        "timestamp": [1005],
    })
    bad_ratings = pd.concat([bad_ratings, duplicate_row], ignore_index=True)

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df)

    err = str(exc_info.value)
    assert "duplicate rating" in err
    assert "userId" in err
    assert "movieId" in err


# ============================================================================
# 5. Referential Integrity / Orphan IDs Tests (AC2)
# ============================================================================


def test_orphan_movie_id_in_ratings_fails(clean_ratings_df, clean_movies_df):
    """Test AC2: Movie ID in ratings that does not exist in movies catalog halts pipeline."""
    bad_ratings = clean_ratings_df.copy()
    # Add a rating for movie 999999, which is not in clean_movies_df
    orphan_row = pd.DataFrame({
        "userId": [1],
        "movieId": [999999],
        "rating": [4.0],
        "timestamp": [1010],
    })
    bad_ratings = pd.concat([bad_ratings, orphan_row], ignore_index=True)

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df)

    err = str(exc_info.value)
    assert "orphan movie ID" in err
    assert "999999" in err


def test_orphan_user_id_in_ratings_fails(clean_ratings_df, clean_movies_df, clean_users_df):
    """Test AC2: User ID in ratings not in users table halts pipeline."""
    bad_ratings = clean_ratings_df.copy()
    # Add rating from user 888888, not in clean_users_df
    orphan_user_row = pd.DataFrame({
        "userId": [888888],
        "movieId": [10],
        "rating": [4.0],
        "timestamp": [1011],
    })
    bad_ratings = pd.concat([bad_ratings, orphan_user_row], ignore_index=True)

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, clean_movies_df, users_df=clean_users_df)

    err = str(exc_info.value)
    assert "orphan user ID" in err
    assert "888888" in err


# ============================================================================
# 6. Multiple Errors Aggregation & Robustness Tests
# ============================================================================


def test_multiple_errors_aggregated_in_exception(clean_ratings_df, clean_movies_df):
    """Test that all errors across all checks are gathered and reported together."""
    bad_ratings = clean_ratings_df.copy()
    bad_movies = clean_movies_df.copy()

    # Issue 1: Missing value in ratings
    bad_ratings.loc[0, "timestamp"] = np.nan
    # Issue 2: Rating out of bounds
    bad_ratings.loc[1, "rating"] = 9.9
    # Issue 3: Duplicate user-movie rating
    dup_row = pd.DataFrame({
        "userId": [2],
        "movieId": [10],
        "rating": [4.0],
        "timestamp": [1050],
    })
    bad_ratings = pd.concat([bad_ratings, dup_row], ignore_index=True)
    # Issue 4: Orphan movie ID
    orphan_row = pd.DataFrame({
        "userId": [3],
        "movieId": [777777],
        "rating": [4.0],
        "timestamp": [1060],
    })
    bad_ratings = pd.concat([bad_ratings, orphan_row], ignore_index=True)
    # Issue 5: Missing value in movies
    bad_movies.loc[0, "genres"] = np.nan

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(bad_ratings, bad_movies)

    err = exc_info.value
    assert len(err.errors) >= 5
    err_str = str(err)
    assert "Data pipeline halted! Found" in err_str
    assert "missing value" in err_str
    assert "outside" in err_str
    assert "duplicate rating" in err_str
    assert "orphan movie ID" in err_str


def test_empty_dataframe_fails():
    """Test that validating empty tables halts with clear error."""
    empty_ratings = pd.DataFrame()
    valid_movies = pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Action"]})

    with pytest.raises(DataValidationError) as exc_info:
        validate_raw_data(empty_ratings, valid_movies)

    assert "empty or None" in str(exc_info.value)


# ============================================================================
# 7. Unit Tests for Standalone Functions
# ============================================================================


def test_check_missing_values_unit():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    issues = check_missing_values(df, "test_table")
    assert len(issues) == 1
    assert "column 'a'" in issues[0]

    clean_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert check_missing_values(clean_df, "test_table") == []


def test_check_rating_range_unit():
    df = pd.DataFrame({"rating": [0.5, 3.0, 5.0]})
    assert check_rating_range(df) == []

    df_invalid = pd.DataFrame({"rating": [-0.5, 5.5]})
    issues = check_rating_range(df_invalid)
    assert len(issues) == 1
    assert "found 2 rating(s) outside" in issues[0]


def test_check_duplicate_ratings_unit():
    df = pd.DataFrame({"userId": [1, 1], "movieId": [10, 10], "rating": [4, 5]})
    issues = check_duplicate_ratings(df)
    assert len(issues) == 1
    assert "duplicate rating" in issues[0]

    df_unique = pd.DataFrame({"userId": [1, 1], "movieId": [10, 20], "rating": [4, 5]})
    assert check_duplicate_ratings(df_unique) == []


def test_check_orphan_ids_unit():
    ratings = pd.DataFrame({"userId": [1], "movieId": [99]})
    movies = pd.DataFrame({"movieId": [1, 2]})
    issues = check_orphan_ids(ratings, movies)
    assert len(issues) == 1
    assert "orphan movie ID" in issues[0]
