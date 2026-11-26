"""Unit tests for recommender evaluation metrics (US 2.2 AC4 & AC5)."""
import pytest
import numpy as np
import pandas as pd

from src.models.metrics import (
    compute_rmse,
    compute_mae,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    evaluate_predictions,
    evaluate_model,
)
from src.models.reference import GlobalMeanModel


def test_compute_rmse_and_mae():
    """Verify standard regression error calculations."""
    y_true = [4.0, 5.0, 3.0, 2.0]
    y_pred = [4.0, 4.0, 3.0, 1.0]

    # Errors: [0, -1, 0, -1] -> sq errors: [0, 1, 0, 1] -> mean 0.5 -> sqrt(0.5) ~ 0.7071
    rmse = compute_rmse(y_true, y_pred)
    assert rmse == pytest.approx(np.sqrt(0.5))

    # Abs errors: [0, 1, 0, 1] -> mean 0.5
    mae = compute_mae(y_true, y_pred)
    assert mae == pytest.approx(0.5)


def test_precision_and_recall_at_k():
    """Verify precision@k and recall@k on known recommendation sets."""
    # Top-5 recommendations: [10, 20, 30, 40, 50]
    # Relevant items: {20, 40, 60, 70} (4 total)
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 60, 70}

    # In top 5: 2 hits (20, 40). Precision@5 = 2 / 5 = 0.4
    p5 = compute_precision_at_k(recommended, relevant, k=5)
    assert p5 == pytest.approx(0.4)

    # Recall@5 = 2 / 4 = 0.5
    r5 = compute_recall_at_k(recommended, relevant, k=5)
    assert r5 == pytest.approx(0.5)

    # In top 2: 1 hit (20). Precision@2 = 1 / 2 = 0.5
    p2 = compute_precision_at_k(recommended, relevant, k=2)
    assert p2 == pytest.approx(0.5)

    # Empty cases
    assert compute_precision_at_k([], relevant, k=5) == 0.0
    assert compute_recall_at_k(recommended, set(), k=5) == 0.0


def test_ndcg_at_k():
    """Verify NDCG@K graded relevance calculation."""
    # Ideal ranking: items rated 5.0 at rank 1, 4.0 at rank 2
    # Recommended: [10, 20]
    ground_truth = {10: 5.0, 20: 4.0}
    # Ideal order -> NDCG should be 1.0
    ndcg_perfect = compute_ndcg_at_k([10, 20], ground_truth, k=2, threshold=3.5)
    assert ndcg_perfect == pytest.approx(1.0)

    # Reversed order -> NDCG < 1.0
    ndcg_reversed = compute_ndcg_at_k([20, 10], ground_truth, k=2, threshold=3.5)
    assert 0.0 < ndcg_reversed < 1.0

    # Irrelevant recommendations
    ndcg_zero = compute_ndcg_at_k([99, 100], ground_truth, k=2, threshold=3.5)
    assert ndcg_zero == 0.0


def test_evaluate_predictions():
    """Verify evaluate_predictions dictionary output."""
    res = evaluate_predictions([4.0, 5.0], [4.0, 3.0])
    assert "rmse" in res
    assert "mae" in res
    assert res["rmse"] == pytest.approx(1.4142, abs=1e-3)
    assert res["mae"] == pytest.approx(1.0)


def test_evaluate_model_integration():
    """AC4: Verify evaluate_model runs end-to-end on a model and returns all metrics."""
    train_df = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2],
        "movie_id": [10, 20, 30, 10, 20, 30],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0],
    })
    test_df = pd.DataFrame({
        "user_id": [1, 2],
        "movie_id": [40, 40],
        "rating": [5.0, 4.0],
    })

    model = GlobalMeanModel().fit(train_df)
    results = evaluate_model(model, test_df=test_df, train_df=train_df, k=2, threshold=4.0)

    assert "rmse" in results
    assert "mae" in results
    assert "precision_at_2" in results
    assert "recall_at_2" in results
    assert "ndcg_at_2" in results
    assert results["rmse"] >= 0.0
