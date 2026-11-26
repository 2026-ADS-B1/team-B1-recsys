"""Unit tests for simple reference baseline models (US 2.2 AC1 & AC6)."""
import pytest
import numpy as np
import pandas as pd

from src.models.reference import GlobalMeanModel, PopularityReferenceModel


@pytest.fixture
def sample_train_data():
    """Create sample ratings training data."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 4],
        "movie_id": [10, 20, 30, 10, 20, 40, 10, 30, 20],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 5.0, 4.0, 4.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })


def test_global_mean_model_fit_and_predict(sample_train_data):
    """Test GlobalMeanModel basic fitting and prediction."""
    model = GlobalMeanModel()
    model.fit(sample_train_data)

    expected_mean = sample_train_data["rating"].mean()
    assert model.global_mean == pytest.approx(expected_mean)

    # Prediction is always global mean
    assert model.predict(user_id=1, movie_id=10) == pytest.approx(expected_mean)
    assert model.predict(user_id=99, movie_id=999) == pytest.approx(expected_mean)

    # Batch prediction
    test_df = pd.DataFrame({"user_id": [1, 2], "movie_id": [10, 20]})
    preds = model.predict_batch(test_df)
    assert len(preds) == 2
    assert np.allclose(preds, expected_mean)


def test_global_mean_model_recommendations(sample_train_data):
    """AC1: Verify GlobalMeanModel produces valid recommendations for users."""
    model = GlobalMeanModel()
    model.fit(sample_train_data)

    # User 1 has seen 10, 20, 30. Unseen should be 40.
    recs = model.recommend(user_id=1, n=2, exclude_seen=True)
    assert not recs.empty
    assert "rank" in recs.columns
    assert "movie_id" in recs.columns
    assert "score" in recs.columns

    # Verify user 1 did not get already rated movies (10, 20, 30)
    assert 10 not in recs["movie_id"].values
    assert 20 not in recs["movie_id"].values
    assert 30 not in recs["movie_id"].values
    assert 40 in recs["movie_id"].values


def test_popularity_reference_model_fit_and_predict(sample_train_data):
    """Test PopularityReferenceModel fitting and prediction."""
    model = PopularityReferenceModel(min_ratings=1)
    model.fit(sample_train_data)

    assert model.ranked_movies is not None
    assert not model.ranked_movies.empty

    # Predict for known movie
    pred = model.predict(user_id=1, movie_id=10)
    assert 1.0 <= pred <= 5.0

    # Unseen movie falls back to global mean
    fallback_pred = model.predict(user_id=1, movie_id=999)
    assert fallback_pred == pytest.approx(sample_train_data["rating"].mean())


def test_popularity_reference_model_recommendations(sample_train_data):
    """AC1: Verify PopularityReferenceModel produces top-N recommendations."""
    model = PopularityReferenceModel(min_ratings=1)
    model.fit(sample_train_data)

    recs = model.recommend(user_id=1, n=2, exclude_seen=True)
    assert len(recs) <= 2
    assert "rank" in recs.columns
    assert "movie_id" in recs.columns
    assert "score" in recs.columns

    # Check descending order
    scores = recs["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_models_config_from_params_yaml():
    """AC6: Verify PopularityReferenceModel loads settings from params.yaml."""
    model = PopularityReferenceModel.from_config("params.yaml")
    assert isinstance(model.min_ratings, int)
    assert model.min_ratings == 50


def test_empty_train_raises():
    """Verify input validation on empty training data."""
    g_model = GlobalMeanModel()
    with pytest.raises(ValueError, match="train_df cannot be empty"):
        g_model.fit(pd.DataFrame())

    p_model = PopularityReferenceModel()
    with pytest.raises(ValueError, match="train_df cannot be empty"):
        p_model.fit(pd.DataFrame())
