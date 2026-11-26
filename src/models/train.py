"""Model training and fair comparison pipeline (US 2.2 AC3, AC4, AC5 & AC6).

Trains both the reference baseline (Popularity) and the advanced model (SVD)
on a 3-way temporal split (Train, Validation, Test), monitors overfitting
with early stopping, evaluates both on the identical unbiased test holdout,
and logs all metrics and artifacts to disk and MLflow / DagsHub.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from src.models.metrics import evaluate_model
from src.models.reference import PopularityReferenceModel
from src.models.split import load_split_params, train_val_test_split_ratings
from src.models.svd import MatrixFactorizationSVDModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(
    ratings_path: str | Path = "data/raw/ml-1m/ratings.dat",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load ratings data from raw MovieLens file or synthetic fallback."""
    path = Path(ratings_path)
    if path.exists():
        logger.info(f"Loading ratings dataset from {path}")
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
    else:
        logger.warning(f"File {path} not found. Generating reproducible fallback data for testing.")
        rows = []
        for u in range(1, 51):
            for m in range(1, 31):
                if (u + m) % 3 != 0:
                    r = 4.0 if (u % 2 == m % 2) else 2.0
                    rows.append((u, m, r, 1000 + u * 10 + m))
        df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])

    if sample_size is not None and len(df) > sample_size:
        logger.info(f"Subsampling dataset to {sample_size:,} interactions for quick evaluation")
        df = df.head(sample_size)

    return df


def run_training_pipeline(
    config_path: str | Path = "params.yaml",
    ratings_path: str | Path = "data/raw/ml-1m/ratings.dat",
    sample_size: Optional[int] = None,
    enable_tracking: bool = True,
    output_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """Execute 3-way training, validation monitoring, comparison and logging pipeline."""
    config_p = Path(config_path)
    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    split_cfg = load_split_params(config_path)
    eval_cfg = cfg.get("models", {}).get("evaluation", {"k": 10, "threshold": 4.0})
    k = int(eval_cfg.get("k", 10))
    threshold = float(eval_cfg.get("threshold", 4.0))

    # 1. Load Data and 3-Way Split (Train / Val / Test)
    df = load_dataset(ratings_path=ratings_path, sample_size=sample_size)
    logger.info(f"Loaded {len(df):,} total ratings from {len(df['user_id'].unique())} users")

    train_df, val_df, test_df = train_val_test_split_ratings(
        df,
        val_size=split_cfg.get("val_size", 0.1),
        test_size=split_cfg.get("test_size", 0.1),
        random_state=split_cfg["random_state"],
        stratified_by_user=split_cfg["stratified_by_user"],
        min_user_ratings=split_cfg["min_user_ratings"],
        temporal=split_cfg.get("temporal", True),
    )
    logger.info(
        f"3-Way Split completed: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,} "
        f"(temporal={split_cfg.get('temporal', True)})"
    )

    # 2. Train and Evaluate Baseline (Popularity)
    logger.info("Training reference baseline: PopularityReferenceModel...")
    pop_model = PopularityReferenceModel.from_config(config_path)
    pop_model.fit(train_df)

    logger.info("Evaluating reference baseline on validation and test sets...")
    pop_val_metrics = evaluate_model(pop_model, test_df=val_df, train_df=train_df, k=k, threshold=threshold)
    pop_test_metrics = evaluate_model(pop_model, test_df=test_df, train_df=train_df, k=k, threshold=threshold)
    logger.info(f"Baseline Test metrics: {pop_test_metrics}")

    # 3. Train and Monitor Advanced Model (SVD with Early Stopping on Validation)
    logger.info("Training advanced model: MatrixFactorizationSVDModel with validation monitoring...")
    svd_model = MatrixFactorizationSVDModel.from_config(config_path)
    svd_model.fit(train_df, val_df=val_df)

    logger.info(
        f"SVD training completed across {len(svd_model.training_history)} epochs. "
        f"Optimal weights restored from epoch {svd_model.best_epoch}."
    )

    # Print epoch progression table
    print("\n" + "=" * 65)
    print("      SVD EPOCH-BY-EPOCH TRAINING & VALIDATION PROGRESSION")
    print("=" * 65)
    print(f" {'Epoch':<7} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Status'}")
    print("-" * 65)
    for ep, (tr, va) in enumerate(zip(svd_model.training_history, svd_model.validation_history), 1):
        status = "★ Best (Saved)" if ep == svd_model.best_epoch else ""
        print(f" {ep:<7} | {tr:<12.4f} | {va:<12.4f} | {status}")
    print("=" * 65)

    logger.info("Evaluating SVD model on final unbiased test set...")
    svd_test_metrics = evaluate_model(svd_model, test_df=test_df, train_df=train_df, k=k, threshold=threshold)
    logger.info(f"SVD Test metrics: {svd_test_metrics}")

    # 4. Fair Comparison Verification (AC4 & AC5)
    beats_baseline = (
        svd_test_metrics["rmse"] < pop_test_metrics["rmse"]
        and svd_test_metrics["mae"] < pop_test_metrics["mae"]
    )
    logger.info(
        f"Verification AC5 (Advanced beats Baseline): {beats_baseline} "
        f"(SVD RMSE {svd_test_metrics['rmse']} vs Pop RMSE {pop_test_metrics['rmse']})"
    )

    # 5. MLflow Tracking Logging (AC3)
    tracking_runs: Dict[str, str] = {}
    if enable_tracking:
        try:
            from src.tracking.runs import log_training

            logger.info("Logging baseline run to MLflow...")
            pop_run_id = log_training(
                pop_model,
                model_name="baseline-popularity",
                params={"min_ratings": pop_model.min_ratings, **split_cfg},
                seed=split_cfg["random_state"],
                metrics=pop_test_metrics,
                example=test_df[["user_id", "movie_id"]].head(5),
            )
            tracking_runs["baseline_run_id"] = pop_run_id

            logger.info("Logging SVD model run to MLflow...")
            svd_params = {
                "n_factors": svd_model.n_factors,
                "learning_rate": svd_model.learning_rate,
                "reg": svd_model.reg,
                "epochs": svd_model.epochs,
                "early_stopping": svd_model.early_stopping,
                "patience": svd_model.patience,
                "best_epoch": svd_model.best_epoch,
                **split_cfg,
            }
            svd_run_id = log_training(
                svd_model,
                model_name="svd-matrix-factorization",
                params=svd_params,
                seed=svd_model.random_state,
                metrics=svd_test_metrics,
                example=test_df[["user_id", "movie_id"]].head(5),
            )
            tracking_runs["svd_run_id"] = svd_run_id
        except Exception as e:
            logger.warning(f"Remote tracking skipped or failed: {e}. Local results are preserved.")

    # 6. Save Persistent Comparison Artifacts (AC3)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "metrics.json"

    comparison_report = {
        "split_config": split_cfg,
        "evaluation_config": eval_cfg,
        "baseline_popularity": {
            "validation_metrics": pop_val_metrics,
            "test_metrics": pop_test_metrics,
            "params": {"min_ratings": pop_model.min_ratings},
        },
        "advanced_svd": {
            "best_epoch": svd_model.best_epoch,
            "training_history": svd_model.training_history,
            "validation_history": svd_model.validation_history,
            "test_metrics": svd_test_metrics,
            "params": {
                "n_factors": svd_model.n_factors,
                "learning_rate": svd_model.learning_rate,
                "reg": svd_model.reg,
                "epochs": svd_model.epochs,
                "early_stopping": svd_model.early_stopping,
                "patience": svd_model.patience,
            },
        },
        "comparison": {
            "svd_beats_baseline_rmse": bool(svd_test_metrics["rmse"] < pop_test_metrics["rmse"]),
            "svd_beats_baseline_mae": bool(svd_test_metrics["mae"] < pop_test_metrics["mae"]),
            "rmse_improvement_percent": round(
                float((pop_test_metrics["rmse"] - svd_test_metrics["rmse"]) / pop_test_metrics["rmse"] * 100), 2
            ),
        },
        "tracking_runs": tracking_runs,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"Fair comparison report saved to {report_file}")
    return comparison_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fairly compare recommender models.")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--ratings", default="data/raw/ml-1m/ratings.dat", help="Path to ratings file")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for fast evaluation")
    parser.add_argument("--no-tracking", action="store_true", help="Disable remote MLflow tracking")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")

    args = parser.parse_args()
    report = run_training_pipeline(
        config_path=args.config,
        ratings_path=args.ratings,
        sample_size=args.sample,
        enable_tracking=not args.no_tracking,
        output_dir=args.output_dir,
    )
    print("\n" + "=" * 65)
    print("           MODEL TEST BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Baseline Popularity Test RMSE -> {report['baseline_popularity']['test_metrics']['rmse']:.4f}")
    print(f"Advanced SVD Model Test RMSE  -> {report['advanced_svd']['test_metrics']['rmse']:.4f}")
    print(f"Test RMSE Improvement         -> {report['comparison']['rmse_improvement_percent']}%")
    print(f"Best SVD Epoch                -> {report['advanced_svd']['best_epoch']} / {report['advanced_svd']['params']['epochs']}")
    print("=" * 65)
