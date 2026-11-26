"""Tests for SVD Hyperparameter Optimization Pipeline (Task 109)."""
import json
from pathlib import Path
import pytest
import pandas as pd
import yaml

from src.models.optimize import generate_param_grid, run_tuning_pipeline


def test_generate_param_grid():
    tuning_cfg = {
        "svd": {
            "n_factors": [10, 20],
            "learning_rate": [0.005, 0.01],
            "reg": [0.02],
            "epochs": 15,
            "patience": 3,
        }
    }
    grid = generate_param_grid(tuning_cfg)
    assert len(grid) == 4
    assert grid[0]["n_factors"] == 10
    assert grid[0]["epochs"] == 15
    assert grid[0]["patience"] == 3

    # Test max_trials truncation
    limited_grid = generate_param_grid(tuning_cfg, max_trials=2)
    assert len(limited_grid) == 2


def test_run_tuning_pipeline_synthetic(tmp_path: Path):
    # 1. Create synthetic ratings
    ratings_file = tmp_path / "ratings.dat"
    rows = []
    # 2 distinct user-item clusters with strong signal
    for u in range(1, 41):
        for m in range(1, 21):
            r = 5.0 if (u <= 20 and m <= 10) or (u > 20 and m > 10) else 1.0
            rows.append(f"{u}::{m}::{r}::{1000 + u * 10 + m}")
    ratings_file.write_text("\n".join(rows), encoding="utf-8")

    # 2. Config file
    config_file = tmp_path / "params.yaml"
    cfg = {
        "models": {
            "split": {
                "val_size": 0.2,
                "test_size": 0.2,
                "random_state": 42,
                "stratified_by_user": True,
                "min_user_ratings": 3,
                "temporal": False,
            },
            "evaluation": {"k": 5, "threshold": 3.5},
            "svd": {"n_factors": 5, "learning_rate": 0.01, "reg": 0.02},
        },
        "tuning": {
            "svd": {
                "n_factors": [4, 8],
                "learning_rate": [0.01, 0.05],
                "reg": [0.02],
                "epochs": 5,
                "patience": 2,
            }
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    out_dir = tmp_path / "reports"

    report = run_tuning_pipeline(
        config_path=config_file,
        ratings_path=ratings_file,
        max_trials=3,
        enable_tracking=False,
        update_params=True,
        output_dir=out_dir,
    )

    assert "best_trial_id" in report
    assert "best_params" in report
    assert "final_test_metrics" in report
    assert len(report["ranking_of_trials"]) == 3
    assert report["ranking_of_trials"][0]["val_rmse"] <= report["ranking_of_trials"][-1]["val_rmse"]

    # Verify report file exists
    assert (out_dir / "optimization_report.json").exists()

    # Verify config updated
    with open(config_file, "r", encoding="utf-8") as f:
        updated_cfg = yaml.safe_load(f)
    assert updated_cfg["models"]["svd"]["n_factors"] == report["best_params"]["n_factors"]
    assert updated_cfg["models"]["svd"]["learning_rate"] == report["best_params"]["learning_rate"]
