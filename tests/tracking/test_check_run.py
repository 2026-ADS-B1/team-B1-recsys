"""`python -m src.tracking.check_run` logs the baseline end to end (US3.2 task 2, AC2).

Runs against a throw-away local MLflow store and a tiny ratings file.
"""

import mlflow
import pandas as pd
import pytest

from src.tracking import check_run
from src.tracking.config import TrackingSettings

RATINGS = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3, 3],
        "movie_id": [10, 20, 10, 20, 10, 30],
        "rating": [5, 3, 4, 2, 5, 1],
        "timestamp": [1, 2, 3, 4, 5, 6],
    }
)


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    settings = TrackingSettings(uri=(tmp_path / "mlruns").as_uri(), experiment="check")
    monkeypatch.setattr(check_run, "load_settings", lambda: settings)
    yield settings
    mlflow.set_tracking_uri("")


def test_baseline_scores_movies_with_enough_ratings():
    model = check_run.PopularityBaseline(min_ratings=2).fit(RATINGS)
    assert set(model.scores) == {10, 20}          # movie 30 has a single rating
    assert model.predict(1, 10) > model.predict(1, 20)
    assert model.predict(1, 30) == 0.0


def test_check_run_logs_one_run_with_model(local_store, tmp_path, capsys):
    path = tmp_path / "ratings.csv"
    RATINGS.to_csv(path, index=False)

    assert check_run.main(["--ratings", str(path), "--min-ratings", "2", "--seed", "7"]) == 0

    run_id = capsys.readouterr().out.split("Run id : ")[1].split()[0]
    mlflow.set_tracking_uri(local_store.uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["seed"] == "7"
    assert run.data.params["min_ratings"] == "2"
    assert run.data.metrics["ranked_movies"] == 2
    assert run.data.tags["purpose"] == "tracking-check"
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert list(model.predict(pd.DataFrame({"movie_id": [30]}))) == [0.0]


def test_missing_data_says_to_run_dvc_pull(tmp_path, capsys):
    assert check_run.main(["--ratings", str(tmp_path / "nope.dat")]) == 1
    assert "dvc pull" in capsys.readouterr().err
