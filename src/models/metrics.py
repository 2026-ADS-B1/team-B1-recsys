"""Evaluation metrics for recommender models (US 2.2 AC4 & AC5).

Computes rating prediction accuracy (RMSE, MAE) and ranking quality
(Precision@K, Recall@K, NDCG@K) for fair model comparisons.
"""
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd


def compute_rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(np.abs(y_t - y_p)))


def compute_precision_at_k(
    recommended_ids: List[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Compute Precision@K: fraction of top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def compute_recall_at_k(
    recommended_ids: List[int],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """Compute Recall@K: fraction of all relevant items retrieved in top-K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def compute_ndcg_at_k(
    recommended_ids: List[int],
    ground_truth_ratings: Dict[int, float],
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Uses graded relevance where ratings >= threshold are considered relevant.
    """
    if k <= 0 or not ground_truth_ratings:
        return 0.0

    top_k = recommended_ids[:k]

    # Calculate DCG@K
    dcg = 0.0
    for idx, item in enumerate(top_k):
        rating = ground_truth_ratings.get(item, 0.0)
        rel = max(0.0, rating - threshold + 1.0) if rating >= threshold else 0.0
        if rel > 0.0:
            dcg += rel / np.log2(idx + 2)  # idx 0 -> log2(2) = 1

    # Calculate IDCG@K (Ideal ranking of all known items for this user)
    all_rel = [
        max(0.0, r - threshold + 1.0)
        for r in ground_truth_ratings.values()
        if r >= threshold
    ]
    all_rel.sort(reverse=True)
    ideal_rel = all_rel[:k]

    if not ideal_rel or sum(ideal_rel) == 0.0:
        return 0.0

    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rel))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_predictions(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> Dict[str, float]:
    """Compute standard rating prediction metrics."""
    return {
        "rmse": round(compute_rmse(y_true, y_pred), 4),
        "mae": round(compute_mae(y_true, y_pred), 4),
    }


def evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    k: int = 10,
    threshold: float = 4.0,
    max_users: Optional[int] = 200,
) -> Dict[str, float]:
    """Comprehensive offline evaluation on test set (AC4: fair comparison).

    Computes:
    - rmse: rating prediction root mean squared error
    - mae: rating prediction mean absolute error
    - precision_at_k: average Precision@K across test users
    - recall_at_k: average Recall@K across test users
    - ndcg_at_k: average NDCG@K across test users

    Args:
        model: Recommender model instance exposing predict_batch and recommend.
        test_df: Hold-out test interactions dataframe.
        train_df: Optional training dataframe (to exclude seen items).
        k: Ranking cutoff length (default 10).
        threshold: Rating threshold to consider an item relevant (default 4.0).
        max_users: Maximum number of users to evaluate for ranking metrics (speeds up testing).

    Returns:
        Dictionary mapping metric names to computed floating point scores.
    """
    if test_df is None or test_df.empty:
        raise ValueError("test_df cannot be empty or None")

    # 1. Rating prediction metrics
    preds = model.predict_batch(test_df)
    y_true = test_df["rating"].to_numpy()
    results = evaluate_predictions(y_true, preds)

    # 2. Ranking metrics across unique test users
    unique_users = test_df["user_id"].unique()
    if max_users is not None and len(unique_users) > max_users:
        np.random.seed(42)
        eval_users = np.random.choice(unique_users, size=max_users, replace=False)
    else:
        eval_users = unique_users

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []

    # Pre-index test user interactions
    test_user_groups = test_df.groupby("user_id")

    for uid in eval_users:
        user_test = test_user_groups.get_group(uid)
        ground_truth_dict = dict(zip(user_test["movie_id"], user_test["rating"]))
        relevant_items = {
            int(mid) for mid, r in ground_truth_dict.items() if r >= threshold
        }

        # Generate top-K recommendations
        recs_df = model.recommend(user_id=int(uid), n=k, exclude_seen=True)
        recommended_items = recs_df["movie_id"].tolist() if not recs_df.empty else []

        precisions.append(compute_precision_at_k(recommended_items, relevant_items, k=k))
        recalls.append(compute_recall_at_k(recommended_items, relevant_items, k=k))
        ndcgs.append(compute_ndcg_at_k(recommended_items, ground_truth_dict, k=k, threshold=threshold))

    results[f"precision_at_{k}"] = round(float(np.mean(precisions)), 4) if precisions else 0.0
    results[f"recall_at_{k}"] = round(float(np.mean(recalls)), 4) if recalls else 0.0
    results[f"ndcg_at_{k}"] = round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0

    return results
