"""Shared MLflow settings and credentials (US3.2 task 1, AC1). No network needed."""

import os

import mlflow
import pytest

from src.tracking import config
from src.tracking.config import TrackingError, TrackingSettings

PARAMS = """
tracking:
  uri: "https://dagshub.com/team/repo.mlflow/"
  experiment: "recommender"
"""

DVC_LOCAL = """['remote "origin"']
    user = dvc-user
    password = dvc-token
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "PARAMS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def params(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(PARAMS, encoding="utf-8")
    return path


@pytest.fixture
def dvc_local(tmp_path):
    path = tmp_path / "config.local"
    path.write_text(DVC_LOCAL, encoding="utf-8")
    return path


def test_settings_come_from_params_yaml(params):
    settings = config.load_settings(params)
    assert settings == TrackingSettings(
        uri="https://dagshub.com/team/repo.mlflow", experiment="recommender"
    )
    assert settings.is_remote


def test_environment_variables_win_over_the_file(params, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-test")
    settings = config.load_settings(params)
    assert settings.uri == "http://localhost:5000"
    assert settings.experiment == "my-test"


def test_missing_server_gives_a_clear_error(tmp_path):
    empty = tmp_path / "params.yaml"
    empty.write_text("serving:\n  min_ratings: 50\n", encoding="utf-8")
    with pytest.raises(TrackingError, match="tracking.uri"):
        config.load_settings(empty)


def test_repository_params_point_to_the_team_server():
    """The real params.yaml is the same for everybody: DagsHub server, shared experiment."""
    settings = config.load_settings("params.yaml")
    assert settings.uri.startswith("https://dagshub.com/") and settings.uri.endswith(".mlflow")
    assert settings.experiment


def test_credentials_are_reused_from_the_dvc_login(dvc_local):
    assert config.resolve_credentials(dvc_local) == ("dvc-user", "dvc-token", str(dvc_local))


def test_environment_credentials_win_over_dvc(dvc_local, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "env-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "env-token")
    user, token, source = config.resolve_credentials(dvc_local)
    assert (user, token, source) == ("env-user", "env-token", "environment variables")


def test_no_credentials_anywhere(tmp_path):
    assert config.resolve_credentials(tmp_path / "missing") is None


def test_configure_sets_server_and_credentials(params, dvc_local, monkeypatch):
    seen = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: seen.setdefault("uri", uri))
    settings = config.configure(config.load_settings(params), dvc_config=dvc_local)
    assert seen["uri"] == settings.uri
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "dvc-user"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "dvc-token"


def test_configure_without_credentials_explains_what_to_do(params, tmp_path, monkeypatch):
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    with pytest.raises(TrackingError, match="README"):
        config.configure(config.load_settings(params), dvc_config=tmp_path / "missing")


def test_local_server_needs_no_credentials(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: seen.setdefault("uri", uri))
    local = TrackingSettings(uri=(tmp_path / "mlruns").as_uri(), experiment="x")
    assert config.configure(local, dvc_config=tmp_path / "missing") == local
    assert seen["uri"] == local.uri
