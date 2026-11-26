"""Integration test for training and comparison pipeline (US 2.2 AC3, AC4 & AC5)."""
import json
import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.models.train import run_training_pipeline


def test_training_pipeline_end_to_end(tmp_path):
    """AC3, AC4 & AC5: Verify end-to-end model training, fair comparison and report persistence."""
    # Create synthetic dataset with distinct user preferences and realistic timestamps
    rng = np.random.RandomState(42)
    rows = []
    # 50 users, 20 movies (2 distinct clusters)
    for u in range(1, 51):
        for m in range(1, 21):
            r = 5.0 if (u <= 25 and m <= 10) or (u > 25 and m > 10) else 1.0
            ts = int(rng.randint(1000, 50000))
            rows.append((u, m, r, ts))

    ratings_df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    ratings_file = tmp_path / "ratings.dat"
    content = ratings_df.to_csv(sep="|", index=False, header=False).replace("|", "::")
    ratings_file.write_text(content, encoding="utf-8")

    # Custom configuration to verify AC6
    config_file = tmp_path / "params.yaml"
    config_dict = {
        "models": {
            "split": {
                "val_size": 0.1,
                "test_size": 0.1,
                "random_state": 42,
                "stratified_by_user": True,
                "min_user_ratings": 5,
                "temporal": True,
            },
            "evaluation": {
                "k": 5,
                "threshold": 4.0,
            },
            "baselines": {
                "popularity": {
                    "min_ratings": 1,
                },
            },
            "svd": {
                "n_factors": 10,
                "learning_rate": 0.01,
                "reg": 0.01,
                "epochs": 25,
                "random_state": 42,
                "early_stopping": True,
                "patience": 5,
            },
        }
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    out_dir = tmp_path / "reports"

    report = run_training_pipeline(
        config_path=config_file,
        ratings_path=ratings_file,
        enable_tracking=False,  # Offline test mode
        output_dir=out_dir,
    )

    # AC3: Report was saved to disk
    report_path = out_dir / "metrics.json"
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    # AC4: Both models evaluated under identical setup on test holdout
    assert "baseline_popularity" in saved_data
    assert "advanced_svd" in saved_data
    assert "test_metrics" in saved_data["baseline_popularity"]
    assert "test_metrics" in saved_data["advanced_svd"]
    assert "validation_history" in saved_data["advanced_svd"]
    assert "best_epoch" in saved_data["advanced_svd"]

    pop_rmse = saved_data["baseline_popularity"]["test_metrics"]["rmse"]
    svd_rmse = saved_data["advanced_svd"]["test_metrics"]["rmse"]

    # AC5: Advanced model beats baseline on this clustered dataset
    assert saved_data["comparison"]["svd_beats_baseline_rmse"] is True
    assert svd_rmse < pop_rmse
