"""Tests for choosing the model with environment variables (issue #26).

MLflow is replaced by a small fake module, so these tests run anywhere
(locally and in CI) without a registry and without installing mlflow.
"""

import sys
import types

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.model_loader import ModelLoadError, load_model
from api.service import RecommenderService


def install_fake_mlflow(monkeypatch, version="3", scores=None):
    """Put a fake `mlflow` module in place. It records which model was asked for."""
    asked = {}

    class FakeModel:
        def predict(self, df):
            # Default: the higher the movie_id, the better the score
            return (
                [scores.get(m, 0.0) for m in df["movie_id"]]
                if scores
                else list(df["movie_id"] * 1.0)
            )

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            asked["name"], asked["alias"] = name, alias
            return types.SimpleNamespace(version=version)

    def fake_load_model(uri):
        asked["uri"] = uri
        return FakeModel()

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = FakeClient
    fake.pyfunc = types.SimpleNamespace(load_model=fake_load_model)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return asked


def test_baseline_is_the_default(monkeypatch, ratings, movies):
    monkeypatch.delenv("MODEL_SOURCE", raising=False)
    model = load_model(ratings, movies, min_ratings=3)
    assert model.info["source"] == "baseline"
    assert model.label == "popularity-baseline"


def test_model_name_and_alias_come_from_environment(monkeypatch, ratings, movies):
    asked = install_fake_mlflow(monkeypatch, version="7")
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.setenv("MODEL_NAME", "my-recommender")
    monkeypatch.setenv("MODEL_ALIAS", "challenger")

    model = load_model(ratings, movies, min_ratings=3)

    assert asked["uri"] == "models:/my-recommender@challenger"
    assert model.info == {
        "source": "mlflow",
        "name": "my-recommender",
        "alias": "challenger",
        "version": "7",
    }


def test_health_shows_the_loaded_model(monkeypatch, ratings, movies):
    install_fake_mlflow(monkeypatch, version="3")
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    service = RecommenderService(ratings, movies, min_ratings=3)

    with TestClient(create_app(service)) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model"]["version"] == "3"
    assert body["model"]["alias"] == "champion"


def test_a_new_model_changes_the_answers_without_code_changes(
    monkeypatch, ratings, movies
):
    # Same code, two "restarts" with a different model version in the registry
    install_fake_mlflow(monkeypatch, version="1", scores={1: 9.0, 2: 8.0})
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    first = RecommenderService(ratings, movies, min_ratings=3)

    install_fake_mlflow(monkeypatch, version="2", scores={20: 9.0, 19: 8.0})
    second = RecommenderService(ratings, movies, min_ratings=3)

    top_first = [m["movie_id"] for m in first.recommend(user_id=1, n=20)]
    top_second = [m["movie_id"] for m in second.recommend(user_id=1, n=20)]
    assert top_first != top_second
    assert second.model_info["version"] == "2"


def test_the_answer_says_which_model_was_used(monkeypatch, ratings, movies):
    install_fake_mlflow(monkeypatch, version="4")
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    service = RecommenderService(ratings, movies, min_ratings=3)
    with TestClient(create_app(service)) as client:
        body = client.post("/recommend", json={"user_id": 1}).json()
    assert body["source"] == "recommender@champion v4"


def test_unknown_source_gives_a_clear_error(monkeypatch, ratings, movies):
    monkeypatch.setenv("MODEL_SOURCE", "magic")
    with pytest.raises(ModelLoadError, match="Unknown MODEL_SOURCE"):
        load_model(ratings, movies, min_ratings=3)


def test_registry_problems_give_a_clear_error(monkeypatch, ratings, movies):
    fake = types.ModuleType("mlflow")

    class BrokenClient:
        def get_model_version_by_alias(self, name, alias):
            raise RuntimeError("registry not reachable")

    fake.MlflowClient = BrokenClient
    fake.pyfunc = types.SimpleNamespace(load_model=lambda uri: None)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")

    with pytest.raises(ModelLoadError, match="models:/recommender@champion"):
        load_model(ratings, movies, min_ratings=3)
