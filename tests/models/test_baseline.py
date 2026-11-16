"""Unit tests for baseline recommendation models."""
import pytest
import pandas as pd
from src.models.baseline import compute_movie_stats


@pytest.fixture
def sample_ratings_df():
    """Create a sample ratings DataFrame with known values for testing."""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2, 3, 1, 2],
        "movie_id": [10, 10, 10, 20, 20, 30, 40, 40],
        "rating": [5, 5, 4, 3, 3, 1, 5, 4],
        "timestamp": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
    })


@pytest.fixture
def sample_movies_df():
    """Create a sample movies metadata DataFrame."""
    return pd.DataFrame({
        "movie_id": [10, 20, 30, 40],
        "title": ["Top Movie (1995)", "Mid Movie (1998)", "Bad Movie (2000)", "Good Movie (2001)"],
        "genres": ["Action|Sci-Fi", "Comedy", "Horror", "Drama"],
    })


def test_compute_movie_stats_basic(sample_ratings_df):
    """Test basic aggregation without metadata."""
    stats = compute_movie_stats(sample_ratings_df, min_ratings=2)

    # Movies with >= 2 ratings are: 10 (3 ratings), 20 (2 ratings), 40 (2 ratings)
    # Movie 30 has only 1 rating, so it should be filtered out
    assert len(stats) == 3
    assert set(stats["movie_id"]) == {10, 20, 40}

    # Verify counts
    movie_10_row = stats[stats["movie_id"] == 10].iloc[0]
    assert movie_10_row["rating_count"] == 3
    assert movie_10_row["mean_rating"] == pytest.approx(4.6667, abs=1e-3)


def test_compute_movie_stats_descending_order(sample_ratings_df):
    """Test AC4: Recommendations must appear in descending order of score."""
    stats = compute_movie_stats(sample_ratings_df, min_ratings=1)

    # Check that score column is monotonically decreasing
    scores = stats["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_compute_movie_stats_min_ratings_filter(sample_ratings_df):
    """Test that min_ratings threshold properly excludes movies with insufficient votes."""
    stats = compute_movie_stats(sample_ratings_df, min_ratings=3)

    # Only movie 10 has 3 ratings
    assert len(stats) == 1
    assert stats.iloc[0]["movie_id"] == 10


def test_compute_movie_stats_with_metadata(sample_ratings_df, sample_movies_df):
    """Test that title and genres are correctly merged when movies_df is provided."""
    stats = compute_movie_stats(sample_ratings_df, min_ratings=2, movies_df=sample_movies_df)

    assert "title" in stats.columns
    assert "genres" in stats.columns

    movie_10 = stats[stats["movie_id"] == 10].iloc[0]
    assert movie_10["title"] == "Top Movie (1995)"
    assert movie_10["genres"] == "Action|Sci-Fi"


def test_compute_movie_stats_missing_columns():
    """Test that ValueError is raised if required columns are missing."""
    invalid_df = pd.DataFrame({"user_id": [1, 2], "invalid_col": [4, 5]})
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_movie_stats(invalid_df)


def test_compute_movie_stats_empty_dataframe():
    """Test that ValueError is raised on empty input."""
    with pytest.raises(ValueError, match="ratings_df cannot be empty"):
        compute_movie_stats(pd.DataFrame())

