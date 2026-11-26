"""Every training becomes a run with params, seed, metrics and model (US3.2 task 2, AC2).

Uses a temporary local MLflow store (no network, nothing sent to DagsHub).
"""

import random

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.tracking import runs
from src.tracking.config import TrackingSettings


class TinyModel:
    """Minimal recommender: score = movie bonus + 0.1 * user id (so users differ)."""

    def __init__(self, bonus):
        self.bonus = bonus

    def predict(self, user_id, movie_id):
        return self.bonus.get(movie_id, 0.0) + 0.1 * max(user_id, 0)


class BatchModel(TinyModel):
    def predict_batch(self, df):
        return [self.predict(u, m) for u, m in zip(df["user_id"], df["movie_id"])]


EXAMPLE = pd.DataFrame({"user_id": [1, 1, 2], "movie_id": [10, 20, 10]})


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    # MLflow 3 refuses a plain folder store unless asked; fine for throw-away test runs
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    settings = TrackingSettings(uri=(tmp_path / "mlruns").as_uri(), experiment="test-exp")
    yield settings
    mlflow.set_tracking_uri("")  # don't leak the temporary store to other tests


def train(local_store, model=None, **overrides):
    kwargs = dict(
        model_name="tiny",
        params={"factors": 8, "split": {"test_size": 0.2}},
        seed=123,
        metrics={"ndcg_at_10": 0.25, "precision_at_10": 0.1},
        example=EXAMPLE,
        settings=local_store,
    )
    kwargs.update(overrides)
    return runs.log_training(model or TinyModel({10: 4.0, 20: 3.0}), **kwargs)


def test_each_training_gets_its_own_id(local_store):
    first, second = train(local_store), train(local_store)
    assert first and second and first != second


def test_settings_seed_and_scores_are_saved_under_the_id(local_store):
    run_id = train(local_store)
    run = mlflow.get_run(run_id)
    assert run.data.params["factors"] == "8"
    assert run.data.params["split.test_size"] == "0.2"  # nested settings are flattened
    assert run.data.params["seed"] == "123"
    assert run.data.metrics == {"ndcg_at_10": 0.25, "precision_at_10": 0.1}
    assert run.data.tags["model_name"] == "tiny"
    assert run.info.run_name == "tiny"
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == "test-exp"


def test_the_trained_model_can_be_loaded_with_the_id(local_store):
    run_id = train(local_store)
    model = runs.load_run_model(run_id, settings=local_store)
    scores = model.predict(EXAMPLE)
    assert list(np.round(scores, 3)) == [4.1, 3.1, 4.2]


def test_models_with_predict_batch_are_supported(local_store):
    run_id = train(local_store, model=BatchModel({10: 1.0}))
    scores = runs.load_run_model(run_id, settings=local_store).predict(EXAMPLE)
    assert list(np.round(scores, 3)) == [1.1, 0.1, 1.2]


def test_model_input_without_user_is_scored_as_a_new_user(local_store):
    run_id = train(local_store)
    model = runs.load_run_model(run_id, settings=local_store)
    scores = model.predict(pd.DataFrame({"movie_id": [10, 20]}))
    assert list(scores) == [4.0, 3.0]


def test_set_seed_makes_random_numbers_repeat():
    runs.set_seed(7)
    first = (random.random(), np.random.rand())
    runs.set_seed(7)
    assert (random.random(), np.random.rand()) == first


def test_flatten_params():
    assert runs.flatten_params({"a": 1, "b": {"c": True, "d": {"e": "x"}}}) == {
        "a": "1",
        "b.c": "True",
        "b.d.e": "x",
    }
