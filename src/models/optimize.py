"""Hyperparameter optimization and model tuning pipeline (Task 109).

Systematically evaluates combinations of SVD hyperparameters (latent factors,
learning rate, L2 regularization) using the validation set with early stopping.
Prevents data leakage by holding out the final test set until the winning
configuration is selected, logs every trial to MLflow / DagsHub, and updates
params.yaml with the best configuration.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from src.models.metrics import evaluate_model
from src.models.split import load_split_params, train_val_test_split_ratings
from src.models.svd import MatrixFactorizationSVDModel
from src.models.train import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_param_grid(
    tuning_cfg: Dict[str, Any],
    max_trials: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate list of parameter combinations from tuning config."""
    svd_tuning = tuning_cfg.get("svd", {})
    epochs = int(svd_tuning.get("epochs", 35))
    patience = int(svd_tuning.get("patience", 4))

    # Support explicit list of trials if provided
    explicit_trials = svd_tuning.get("trials")
    if explicit_trials and isinstance(explicit_trials, list):
        grid = []
        for t in explicit_trials:
            grid.append({
                "n_factors": int(t.get("n_factors", 20)),
                "learning_rate": float(t.get("learning_rate", 0.005)),
                "reg": float(t.get("reg", 0.02)),
                "epochs": int(t.get("epochs", epochs)),
                "patience": int(t.get("patience", patience)),
                "early_stopping": True,
                "random_state": 42,
            })
        if max_trials is not None and len(grid) > max_trials:
            grid = grid[:max_trials]
        return grid

    factors_list = svd_tuning.get("n_factors", [10, 20, 30])
    lr_list = svd_tuning.get("learning_rate", [0.002, 0.005, 0.01])
    reg_list = svd_tuning.get("reg", [0.01, 0.02, 0.05])

    # Normalize to lists
    if not isinstance(factors_list, list):
        factors_list = [factors_list]
    if not isinstance(lr_list, list):
        lr_list = [lr_list]
    if not isinstance(reg_list, list):
        reg_list = [reg_list]

    grid = []
    for factors, lr, reg in itertools.product(factors_list, lr_list, reg_list):
        grid.append({
            "n_factors": int(factors),
            "learning_rate": float(lr),
            "reg": float(reg),
            "epochs": epochs,
            "patience": patience,
            "early_stopping": True,
            "random_state": 42,
        })

    if max_trials is not None and len(grid) > max_trials:
        grid = grid[:max_trials]

    return grid


def run_tuning_pipeline(
    config_path: str | Path = "params.yaml",
    ratings_path: str | Path = "data/raw/ml-1m/ratings.dat",
    sample_size: Optional[int] = None,
    max_trials: Optional[int] = None,
    enable_tracking: bool = True,
    update_params: bool = True,
    output_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """Execute hyperparameter optimization, MLflow logging and model selection."""
    config_p = Path(config_path)
    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    split_cfg = load_split_params(config_path)
    tuning_cfg = cfg.get("tuning", {})
    eval_cfg = cfg.get("models", {}).get("evaluation", {"k": 10, "threshold": 4.0})
    k = int(eval_cfg.get("k", 10))
    threshold = float(eval_cfg.get("threshold", 4.0))

    param_combinations = generate_param_grid(tuning_cfg, max_trials=max_trials)
    logger.info(f"Generated {len(param_combinations)} hyperparameter configurations to explore")

    # 1. 3-Way Temporal Data Split (Train / Val / Test)
    df = load_dataset(ratings_path=ratings_path, sample_size=sample_size)
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
        f"Data prepared: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}"
    )

    trials_results: List[Dict[str, Any]] = []
    best_model: Optional[MatrixFactorizationSVDModel] = None
    best_val_rmse = float("inf")
    best_trial_info: Optional[Dict[str, Any]] = None

    # 2. Iterate and evaluate each combination on Validation Set
    print("\n" + "=" * 80)
    print("                    HYPERPARAMETER SEARCH TRIALS")
    print("=" * 80)
    print(f" {'Trial':<6} | {'Factors':<8} | {'LR':<7} | {'Reg':<7} | {'Best Ep':<8} | {'Val RMSE':<10} | {'Val MAE':<10}")
    print("-" * 80)

    for idx, params in enumerate(param_combinations, 1):
        model = MatrixFactorizationSVDModel(
            n_factors=params["n_factors"],
            learning_rate=params["learning_rate"],
            reg=params["reg"],
            epochs=params["epochs"],
            random_state=params["random_state"],
            early_stopping=params["early_stopping"],
            patience=params["patience"],
        )
        model.fit(train_df, val_df=val_df)
        val_metrics = evaluate_model(model, test_df=val_df, train_df=train_df, k=k, threshold=threshold)

        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics["mae"]

        print(
            f" {idx:<6} | {params['n_factors']:<8} | {params['learning_rate']:<7.4f} | "
            f"{params['reg']:<7.3f} | {model.best_epoch:<8} | {val_rmse:<10.4f} | {val_mae:<10.4f}"
        )

        trial_record = {
            "trial_id": idx,
            "params": params,
            "best_epoch": model.best_epoch,
            "val_metrics": val_metrics,
            "training_history": model.training_history,
            "validation_history": model.validation_history,
            "mlflow_run_id": None,
        }

        # 3. Log trial to MLflow / DagsHub
        if enable_tracking:
            try:
                from src.tracking.runs import log_training

                trial_name = f"svd-f{params['n_factors']}-lr{params['learning_rate']}-reg{params['reg']}"
                run_params = {
                    **params,
                    "best_epoch": model.best_epoch,
                    "split_val_size": split_cfg.get("val_size", 0.1),
                    "split_temporal": split_cfg.get("temporal", True),
                }
                run_id = log_training(
                    model,
                    model_name=trial_name,
                    params=run_params,
                    seed=params["random_state"],
                    metrics=val_metrics,
                    example=val_df[["user_id", "movie_id"]].head(5),
                    tags={"purpose": "hyperparameter-optimization", "trial": str(idx)},
                )
                trial_record["mlflow_run_id"] = run_id
            except Exception as e:
                logger.warning(f"MLflow logging failed for trial {idx}: {e}")

        trials_results.append(trial_record)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_trial_info = trial_record

    print("=" * 80)
    logger.info(
        f"Optimization complete! Best trial #{best_trial_info['trial_id']} "
        f"with Val RMSE: {best_val_rmse:.4f} (params: {best_trial_info['params']})"
    )

    # 4. Final Unbiased Evaluation of Best Model on Held-Out Test Set
    logger.info("Evaluating winning model on final held-out Test Set...")
    test_metrics = evaluate_model(best_model, test_df=test_df, train_df=train_df, k=k, threshold=threshold)
    logger.info(f"Final Test Metrics for Best Model: {test_metrics}")

    # 5. Save Optimization Report Artifact
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "optimization_report.json"

    # Sort trials by validation rmse
    sorted_trials = sorted(trials_results, key=lambda t: t["val_metrics"]["rmse"])

    report = {
        "best_trial_id": best_trial_info["trial_id"],
        "best_params": best_trial_info["params"],
        "best_val_metrics": best_trial_info["val_metrics"],
        "final_test_metrics": test_metrics,
        "ranking_of_trials": [
            {
                "trial_id": t["trial_id"],
                "n_factors": t["params"]["n_factors"],
                "learning_rate": t["params"]["learning_rate"],
                "reg": t["params"]["reg"],
                "best_epoch": t["best_epoch"],
                "val_rmse": t["val_metrics"]["rmse"],
                "val_mae": t["val_metrics"]["mae"],
                "mlflow_run_id": t["mlflow_run_id"],
            }
            for t in sorted_trials
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Optimization report saved to {report_path}")

    # 6. Update params.yaml with winning parameters if requested
    if update_params and config_p.exists():
        logger.info(f"Updating {config_p} with optimal SVD hyperparameters...")
        cfg.setdefault("models", {}).setdefault("svd", {})
        cfg["models"]["svd"]["n_factors"] = best_trial_info["params"]["n_factors"]
        cfg["models"]["svd"]["learning_rate"] = best_trial_info["params"]["learning_rate"]
        cfg["models"]["svd"]["reg"] = best_trial_info["params"]["reg"]
        with open(config_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info("params.yaml successfully updated with winning configuration")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SVD Recommender")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--ratings", default="data/raw/ml-1m/ratings.dat", help="Path to ratings.dat")
    parser.add_argument("--sample", type=int, default=30000, help="Subsample ratings for faster tuning")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit number of trial combinations")
    parser.add_argument("--no-tracking", action="store_true", help="Disable remote MLflow logging")
    parser.add_argument("--no-update-params", action="store_true", help="Do not update params.yaml")

    args = parser.parse_args()

    report = run_tuning_pipeline(
        config_path=args.config,
        ratings_path=args.ratings,
        sample_size=args.sample,
        max_trials=args.max_trials,
        enable_tracking=not args.no_tracking,
        update_params=not args.no_update_params,
    )

    print("\n" + "=" * 65)
    print("           OPTIMAL HYPERPARAMETERS SELECTED")
    print("=" * 65)
    best = report["best_params"]
    print(f" Best Latent Factors  -> {best['n_factors']}")
    print(f" Best Learning Rate   -> {best['learning_rate']}")
    print(f" Best Regularization  -> {best['reg']}")
    print(f" Validation RMSE      -> {report['best_val_metrics']['rmse']:.4f}")
    print(f" Final Test RMSE      -> {report['final_test_metrics']['rmse']:.4f}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
