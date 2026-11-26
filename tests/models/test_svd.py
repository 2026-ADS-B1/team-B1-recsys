"""Unit tests for Matrix Factorization / SVD recommender model (US 2.2 AC2 & AC6)."""
import pytest
import numpy as np
import pandas as pd

from src.models.svd import MatrixFactorizationSVDModel


@pytest.fixture
def sample_ratings_df():
    """Create a structured synthetic dataset where user preferences can be learned."""
    # Users 1 and 2 like Sci-Fi (movies 101, 102), dislike Romance (movies 201, 202)
    # Users 3 and 4 like Romance (movies 201, 202), dislike Sci-Fi (movies 101, 102)
    data = [
        (1, 101, 5.0), (1, 102, 4.5), (1, 201, 1.5), (1, 202, 2.0),
        (2, 101, 4.5), (2, 102, 5.0), (2, 201, 2.0), (2, 202, 1.0),
        (3, 101, 1.0), (3, 102, 2.0), (3, 201, 5.0), (3, 202, 4.5),
        (4, 101, 2.0), (4, 102, 1.5), (4, 201, 4.5), (4, 202, 5.0),
        (5, 101, 4.0), (5, 201, 4.0), (5, 301, 3.5), (5, 302, 3.0),
    ]
    return pd.DataFrame(data, columns=["user_id", "movie_id", "rating"])


def test_svd_init_and_config():
    """AC6: Test initialization and configuration loading from params.yaml."""
    model = MatrixFactorizationSVDModel.from_config("params.yaml")
    assert model.n_factors == 20
    assert model.learning_rate == pytest.approx(0.005)
    assert model.reg == pytest.approx(0.02)
    assert model.epochs == 15
    assert model.random_state == 42


def test_svd_fit_learning_convergence(sample_ratings_df):
    """AC2: Test that SVD model trains and objective error decreases over epochs."""
    model = MatrixFactorizationSVDModel(
        n_factors=5,
        learning_rate=0.05,
        reg=0.01,
        epochs=30,
        random_state=42,
    )
    model.fit(sample_ratings_df)

    assert len(model.training_history) == 30
    # Training RMSE at the end should be substantially lower than at start
    assert model.training_history[-1] < model.training_history[0]
    # Final training RMSE on this clean dataset should be under 1.0
    assert model.training_history[-1] < 1.0


def test_svd_predict_known_and_bounds(sample_ratings_df):
    """Test rating predictions for known user-item interactions stay within [1.0, 5.0]."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=10, random_state=42)
    model.fit(sample_ratings_df)

    pred = model.predict(user_id=1, movie_id=101)
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 5.0


def test_svd_predict_cold_start(sample_ratings_df):
    """Test graceful fallback behavior on cold-start users or items."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=5, random_state=42)
    model.fit(sample_ratings_df)

    global_mean = sample_ratings_df["rating"].mean()

    # Known user, unknown movie: mu + b_u
    pred_user_known = model.predict(user_id=1, movie_id=9999)
    assert 1.0 <= pred_user_known <= 5.0

    # Unknown user, known movie: mu + b_i
    pred_movie_known = model.predict(user_id=9999, movie_id=101)
    assert 1.0 <= pred_movie_known <= 5.0

    # Both unknown: fallback to global mean
    pred_both_unknown = model.predict(user_id=9999, movie_id=9999)
    assert pred_both_unknown == pytest.approx(global_mean)


def test_svd_predict_batch(sample_ratings_df):
    """Test batch prediction outputs numpy array matching individual predict calls."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=5, random_state=42)
    model.fit(sample_ratings_df)

    test_pairs = pd.DataFrame({
        "user_id": [1, 2, 999],
        "movie_id": [101, 201, 101],
    })

    batch_preds = model.predict_batch(test_pairs)
    assert len(batch_preds) == 3

    for idx, row in test_pairs.iterrows():
        expected = model.predict(int(row["user_id"]), int(row["movie_id"]))
        assert batch_preds[idx] == pytest.approx(expected)

    # Empty test batch
    assert len(model.predict_batch(pd.DataFrame())) == 0


def test_svd_recommendations_structure_and_order(sample_ratings_df):
    """AC2: Verify recommendation dataframe structure and score ranking."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=15, random_state=42)
    model.fit(sample_ratings_df)

    recs = model.recommend(user_id=1, n=3, exclude_seen=True)
    assert not recs.empty
    assert list(recs.columns) == ["rank", "movie_id", "score"]
    assert len(recs) <= 3
    assert list(recs["rank"]) == list(range(1, len(recs) + 1))

    # Scores must be sorted descending
    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_svd_recommendations_exclude_seen(sample_ratings_df):
    """AC2: Verify already seen movies are properly excluded by default."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=15, random_state=42)
    model.fit(sample_ratings_df)

    # User 1 has seen 101, 102, 201, 202. Unseen in dataset are 301, 302.
    recs_excluded = model.recommend(user_id=1, n=5, exclude_seen=True)
    for seen_id in [101, 102, 201, 202]:
        assert seen_id not in recs_excluded["movie_id"].values

    # With exclude_seen=False, seen movies can appear in recommendations
    recs_included = model.recommend(user_id=1, n=5, exclude_seen=False)
    assert any(seen_id in recs_included["movie_id"].values for seen_id in [101, 102])


def test_svd_recommendations_cold_start_user(sample_ratings_df):
    """AC2: Verify recommendations work for unknown / new users."""
    model = MatrixFactorizationSVDModel(n_factors=5, epochs=10, random_state=42)
    model.fit(sample_ratings_df)

    recs = model.recommend(user_id=9999, n=3, exclude_seen=True)
    assert len(recs) == 3
    assert list(recs.columns) == ["rank", "movie_id", "score"]
    assert recs["score"].is_monotonic_decreasing


def test_svd_reproducibility(sample_ratings_df):
    """Verify that models with same random_state produce identical predictions."""
    model1 = MatrixFactorizationSVDModel(n_factors=5, epochs=10, random_state=42)
    model1.fit(sample_ratings_df)

    model2 = MatrixFactorizationSVDModel(n_factors=5, epochs=10, random_state=42)
    model2.fit(sample_ratings_df)

    p1 = model1.predict(user_id=1, movie_id=101)
    p2 = model2.predict(user_id=1, movie_id=101)
    assert p1 == pytest.approx(p2)


def test_svd_validation_errors(sample_ratings_df):
    """Verify appropriate exceptions for invalid states and data."""
    model = MatrixFactorizationSVDModel()

    # Predict before fit
    with pytest.raises(RuntimeError, match="Model must be fitted"):
        model.predict(1, 101)

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        model.predict_batch(pd.DataFrame({"user_id": [1], "movie_id": [101]}))

    # Empty training data
    with pytest.raises(ValueError, match="train_df cannot be empty"):
        model.fit(pd.DataFrame())

    # Missing columns
    with pytest.raises(ValueError, match="missing required columns"):
        model.fit(pd.DataFrame({"user_id": [1], "movie_id": [101]}))
