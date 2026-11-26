"""The one-command launcher `python -m src.tracking.ui` (US3.2 task 1, AC1). No network needed."""

import types

import mlflow
import pytest

from src.tracking import ui
from src.tracking.config import TrackingError, TrackingSettings

SETTINGS = TrackingSettings(uri="https://dagshub.com/team/repo.mlflow", experiment="recommender")


class FakeClient:
    """Stands in for mlflow.MlflowClient: either answers or fails like the server would."""

    def __init__(self, experiment_id="7", error=None):
        self.experiment_id = experiment_id
        self.error = error

    def __call__(self):
        return self

    def search_experiments(self, max_results=1):
        if self.error:
            raise self.error
        return []

    def get_experiment_by_name(self, name):
        if self.experiment_id is None:
            return None
        return types.SimpleNamespace(experiment_id=self.experiment_id, name=name)


@pytest.fixture
def launcher(monkeypatch):
    """Run main() without touching the real server or browser; returns what was opened."""
    opened = []
    monkeypatch.setattr(ui, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(ui, "configure", lambda settings: settings)
    monkeypatch.setattr(ui.webbrowser, "open", opened.append)
    return opened


def test_opens_the_shared_experiment_in_the_browser(launcher, monkeypatch, capsys):
    monkeypatch.setattr(mlflow, "MlflowClient", FakeClient(experiment_id="7"))
    assert ui.main([]) == 0
    assert launcher == ["https://dagshub.com/team/repo.mlflow/#/experiments/7"]
    assert "recommender" in capsys.readouterr().out


def test_no_browser_only_prints_the_address(launcher, monkeypatch, capsys):
    monkeypatch.setattr(mlflow, "MlflowClient", FakeClient(experiment_id="7"))
    assert ui.main(["--no-browser"]) == 0
    assert launcher == []
    assert "#/experiments/7" in capsys.readouterr().out


def test_experiment_not_created_yet_opens_the_experiments_list(launcher, monkeypatch, capsys):
    monkeypatch.setattr(mlflow, "MlflowClient", FakeClient(experiment_id=None))
    assert ui.main([]) == 0
    assert launcher == ["https://dagshub.com/team/repo.mlflow/#/experiments"]
    assert "no runs yet" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error, hint",
    [
        (Exception("API request failed with error code 401 Unauthorized"), "credentials"),
        (Exception("403 Forbidden"), "collaborator"),
        (Exception("Max retries exceeded with url"), "internet connection"),
    ],
)
def test_server_problems_are_explained_not_dumped(launcher, monkeypatch, capsys, error, hint):
    monkeypatch.setattr(mlflow, "MlflowClient", FakeClient(error=error))
    assert ui.main([]) == 1
    assert launcher == []
    assert hint in capsys.readouterr().err


def test_configuration_errors_stop_with_exit_code_1(monkeypatch, capsys):
    def broken():
        raise TrackingError("No MLflow server configured.")

    monkeypatch.setattr(ui, "load_settings", broken)
    assert ui.main([]) == 1
    assert "No MLflow server configured" in capsys.readouterr().err
