"""Unit tests for baseline recommendation models."""
import pytest
import pandas as pd
from src.models.baseline import (
    compute_movie_stats,
    get_user_recommendations,
    generate_batch_recommendations,
)


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


# ---------------------------------------------------------------------------
# Task 2 Tests: Top-N Batch Recommendations without Duplicates (US 2.1)
# ---------------------------------------------------------------------------

@pytest.fixture
def ranked_movies_fixture():
    """Pre-computed ranked movies DataFrame for recommendation tests."""
    return pd.DataFrame({
        "movie_id": [101, 102, 103, 104, 105],
        "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
        "genres": ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"],
        "rating_count": [100, 80, 60, 40, 20],
        "mean_rating": [4.8, 4.5, 4.2, 3.9, 3.5],
        "score": [4.75, 4.45, 4.15, 3.85, 3.45],
    })


def test_get_user_recommendations_exactly_n(ranked_movies_fixture):
    """AC2: Given a user has >= N eligible items, exactly N recommendations are returned."""
    recs = get_user_recommendations(user_id=1, ranked_movies_df=ranked_movies_fixture, n=3)

    assert len(recs) == 3
    assert list(recs["rank"]) == [1, 2, 3]
    assert list(recs["movie_id"]) == [101, 102, 103]


def test_get_user_recommendations_fewer_than_n(ranked_movies_fixture):
    """AC3: Given fewer than N eligible items, return all available without duplicates or artificial items."""
    small_catalog = ranked_movies_fixture.head(2)
    recs = get_user_recommendations(user_id=1, ranked_movies_df=small_catalog, n=5)

    assert len(recs) == 2
    assert list(recs["rank"]) == [1, 2]
    assert list(recs["movie_id"]) == [101, 102]


def test_get_user_recommendations_excludes_already_rated(ranked_movies_fixture):
    """Test that already rated movies are excluded from candidate recommendations."""
    # User 1 has already watched/rated movie 101 and 102
    user_ratings = pd.DataFrame({
        "user_id": [1, 1, 2],
        "movie_id": [101, 102, 101],
        "rating": [5, 4, 3],
    })

    recs = get_user_recommendations(
        user_id=1,
        ranked_movies_df=ranked_movies_fixture,
        ratings_df=user_ratings,
        n=3,
        exclude_rated=True,
    )

    # 101 and 102 should NOT be in the recommendations
    recommended_ids = recs["movie_id"].tolist()
    assert 101 not in recommended_ids
    assert 102 not in recommended_ids
    # Should get the next best: 103, 104, 105
    assert recommended_ids == [103, 104, 105]
    assert list(recs["rank"]) == [1, 2, 3]


def test_get_user_recommendations_no_duplicates(ranked_movies_fixture):
    """AC3: Verify that duplicate movie records in candidates are deduped."""
    duplicate_catalog = pd.concat([ranked_movies_fixture, ranked_movies_fixture.head(2)], ignore_index=True)
    recs = get_user_recommendations(user_id=1, ranked_movies_df=duplicate_catalog, n=5)

    assert len(recs) == 5
    assert len(recs["movie_id"].unique()) == 5


def test_get_user_recommendations_descending_order(ranked_movies_fixture):
    """AC4: Recommendations must appear in descending order of predicted score."""
    shuffled = ranked_movies_fixture.sample(frac=1, random_state=42)
    recs = get_user_recommendations(user_id=1, ranked_movies_df=shuffled, n=4)

    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_generate_batch_recommendations(ranked_movies_fixture):
    """AC1 & AC2: Test batch recommendations for multiple users."""
    ratings_history = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [101, 102],
        "rating": [5, 4],
    })

    user_ids = [1, 2, 3]
    batch_recs = generate_batch_recommendations(
        user_ids=user_ids,
        ranked_movies_df=ranked_movies_fixture,
        ratings_df=ratings_history,
        n=2,
    )

    assert "user_id" in batch_recs.columns
    assert "rank" in batch_recs.columns
    assert set(batch_recs["user_id"].unique()) == {1, 2, 3}

    # Each user gets exactly 2 recs
    for uid in user_ids:
        user_slice = batch_recs[batch_recs["user_id"] == uid]
        assert len(user_slice) == 2
        assert list(user_slice["rank"]) == [1, 2]

    # User 1 had seen 101 -> should get 102, 103
    u1_recs = batch_recs[batch_recs["user_id"] == 1]["movie_id"].tolist()
    assert u1_recs == [102, 103]

    # User 2 had seen 102 -> should get 101, 103
    u2_recs = batch_recs[batch_recs["user_id"] == 2]["movie_id"].tolist()
    assert u2_recs == [101, 103]


def test_get_user_recommendations_empty():
    """Test behavior with empty candidate dataframe."""
    empty_df = pd.DataFrame()
    recs = get_user_recommendations(user_id=1, ranked_movies_df=empty_df, n=5)
    assert recs.empty


