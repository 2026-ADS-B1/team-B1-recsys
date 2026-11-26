"""Unit tests for reproducible train/test split (US 2.2 AC4 & AC6)."""
import pytest
import numpy as np
import pandas as pd

from src.models.split import train_test_split_ratings, load_split_params


@pytest.fixture
def sample_ratings_dataset():
    """Create a synthetic dataset with 10 users, each rating 10 movies."""
    records = []
    for uid in range(1, 11):
        for mid in range(1, 11):
            records.append({
                "user_id": uid,
                "movie_id": mid,
                "rating": float((uid + mid) % 5 + 1),
                "timestamp": 1000 + uid * 10 + mid,
            })
    return pd.DataFrame(records)


def test_train_test_split_reproducibility(sample_ratings_dataset):
    """AC4: Verify that identical random seeds produce identical splits."""
    train1, test1 = train_test_split_ratings(sample_ratings_dataset, test_size=0.2, random_state=42)
    train2, test2 = train_test_split_ratings(sample_ratings_dataset, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_train_test_split_different_seeds(sample_ratings_dataset):
    """Verify that different random seeds produce different partitions."""
    train1, test1 = train_test_split_ratings(sample_ratings_dataset, test_size=0.2, random_state=42)
    train2, test2 = train_test_split_ratings(sample_ratings_dataset, test_size=0.2, random_state=999)

    # Different seeds should sample different test rows
    test1_tuples = set(zip(test1["user_id"], test1["movie_id"]))
    test2_tuples = set(zip(test2["user_id"], test2["movie_id"]))
    assert test1_tuples != test2_tuples


def test_train_test_split_no_data_leakage(sample_ratings_dataset):
    """Verify that train and test have no overlapping pairs and sum to full size."""
    train, test = train_test_split_ratings(sample_ratings_dataset, test_size=0.2, random_state=42)

    assert len(train) + len(test) == len(sample_ratings_dataset)

    train_pairs = set(zip(train["user_id"], train["movie_id"]))
    test_pairs = set(zip(test["user_id"], test["movie_id"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_train_test_split_proportions(sample_ratings_dataset):
    """Verify that the test set size is approximately the requested ratio."""
    test_size = 0.2
    train, test = train_test_split_ratings(sample_ratings_dataset, test_size=test_size, random_state=42)

    actual_ratio = len(test) / len(sample_ratings_dataset)
    assert actual_ratio == pytest.approx(test_size, abs=0.05)


def test_train_test_split_stratified_by_user(sample_ratings_dataset):
    """Verify that users with >= min_user_ratings are represented in both train and test."""
    train, test = train_test_split_ratings(
        sample_ratings_dataset,
        test_size=0.2,
        random_state=42,
        stratified_by_user=True,
        min_user_ratings=5,
    )

    train_users = set(train["user_id"].unique())
    test_users = set(test["user_id"].unique())

    # All users in this fixture have 10 ratings >= 5, so all must be in both
    assert train_users == set(range(1, 11))
    assert test_users == set(range(1, 11))


def test_train_test_split_cold_start_users_kept_in_train():
    """Verify users with fewer than min_user_ratings stay in train to avoid cold-start evaluation."""
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2],  # User 1 has 5, User 2 has 2
        "movie_id": [10, 20, 30, 40, 50, 10, 20],
        "rating": [5, 4, 3, 4, 5, 3, 2],
    })

    train, test = train_test_split_ratings(df, test_size=0.2, random_state=42, min_user_ratings=5)

    # User 2 should NOT be in test
    assert 2 not in test["user_id"].values
    assert 2 in train["user_id"].values


def test_train_test_split_invalid_inputs():
    """Verify validation on malformed inputs."""
    with pytest.raises(ValueError, match="ratings_df cannot be empty"):
        train_test_split_ratings(pd.DataFrame())

    invalid_df = pd.DataFrame({"bad_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        train_test_split_ratings(invalid_df)

    valid_df = pd.DataFrame({"user_id": [1], "movie_id": [1], "rating": [5]})
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        train_test_split_ratings(valid_df, test_size=1.5)


def test_load_split_params():
    """AC6: Verify loading split configuration from params.yaml."""
    cfg = load_split_params("params.yaml")
    assert "test_size" in cfg
    assert "random_state" in cfg
    assert isinstance(cfg["test_size"], float)
    assert isinstance(cfg["random_state"], int)


def test_train_test_split_temporal_ordering():
    """Verify temporal split places chronologically latest ratings into test."""
    # User 1 has 5 ratings with timestamps 100, 200, 300, 400, 500
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1],
        "movie_id": [10, 20, 30, 40, 50],
        "rating": [5, 4, 3, 4, 5],
        "timestamp": [100, 200, 300, 400, 500],
    })

    train, test = train_test_split_ratings(
        df,
        test_size=0.2,
        stratified_by_user=True,
        min_user_ratings=5,
        temporal=True,
    )

    # Test size is 1 item (20% of 5). It must be the latest timestamp (500, movie 50)
    assert len(test) == 1
    assert test.iloc[0]["movie_id"] == 50
    assert test.iloc[0]["timestamp"] == 500

    # Train timestamps must all be less than test timestamp
    assert train["timestamp"].max() < test["timestamp"].min()


def test_train_val_test_split_proportions_and_temporal():
    """Verify 3-way split divides data correctly across Train, Validation and Test sets."""
    from src.models.split import train_val_test_split_ratings

    # Create 10 ratings for user 1 with timestamps 10..100
    df = pd.DataFrame({
        "user_id": [1] * 10,
        "movie_id": list(range(1, 11)),
        "rating": [4.0] * 10,
        "timestamp": [i * 10 for i in range(1, 11)],
    })

    train, val, test = train_val_test_split_ratings(
        df,
        val_size=0.1,
        test_size=0.1,
        stratified_by_user=True,
        min_user_ratings=5,
        temporal=True,
    )

    # 10% of 10 is 1 item in test, 1 item in val, 8 in train
    assert len(test) == 1
    assert len(val) == 1
    assert len(train) == 8

    # Temporal order: train < val < test
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert test.iloc[0]["timestamp"] == 100
    assert val.iloc[0]["timestamp"] == 90
