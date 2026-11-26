"""Same MLflow setup for every team member (US3.2, task 1).

The server address and the experiment name live in the `tracking` section of
params.yaml, so everyone logs to the same place. Environment variables win over
the file: MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME (the names MLflow itself
uses).

Credentials are never stored in the repository. For a remote server (http/https)
they are taken from:
  1. MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD, if both are set, or
  2. the DVC login of the DagsHub remote in .dvc/config.local (the same DagsHub user
     and token you already configured for `dvc pull`, see README -> Data).
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = "params.yaml"
DEFAULT_DVC_LOCAL_CONFIG = ".dvc/config.local"
DVC_REMOTE = "origin"


class TrackingError(Exception):
    """The tracking settings are missing or wrong, or the server can't be used."""


@dataclass(frozen=True)
class TrackingSettings:
    uri: str
    experiment: str

    @property
    def is_remote(self) -> bool:
        return self.uri.startswith(("http://", "https://"))


def load_settings(path: str | Path | None = None) -> TrackingSettings:
    """Read the `tracking` section of params.yaml (environment variables win)."""
    path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    tracking: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            tracking = (yaml.safe_load(f) or {}).get("tracking") or {}

    uri = os.getenv("MLFLOW_TRACKING_URI") or tracking.get("uri")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME") or tracking.get("experiment")
    if not uri:
        raise TrackingError(
            f"No MLflow server configured. Add 'tracking.uri' to {path} "
            "or set MLFLOW_TRACKING_URI. Run the command from the repository root."
        )
    if not experiment:
        raise TrackingError(
            f"No experiment name configured. Add 'tracking.experiment' to {path} "
            "or set MLFLOW_EXPERIMENT_NAME."
        )
    return TrackingSettings(uri=str(uri).rstrip("/"), experiment=str(experiment))


def _credentials_from_dvc(config_path: str | Path) -> tuple[str, str] | None:
    """User and token of the DVC remote 'origin' in .dvc/config.local, if present."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except configparser.Error:
        return None
    for section in parser.sections():
        # DVC writes the header as ['remote "origin"'] -> section name 'remote "origin"'
        if section.strip().strip("'") == f'remote "{DVC_REMOTE}"':
            user = parser[section].get("user")
            password = parser[section].get("password")
            if user and password:
                return user, password
    return None


def resolve_credentials(
    dvc_config: str | Path = DEFAULT_DVC_LOCAL_CONFIG,
) -> tuple[str, str, str] | None:
    """Return (user, token, where it came from), or None if there are none."""
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if user and token:
        return user, token, "environment variables"
    from_dvc = _credentials_from_dvc(dvc_config)
    if from_dvc:
        return from_dvc[0], from_dvc[1], str(dvc_config)
    return None


def configure(
    settings: TrackingSettings | None = None,
    dvc_config: str | Path = DEFAULT_DVC_LOCAL_CONFIG,
) -> TrackingSettings:
    """Point MLflow at the team's server, with the user's credentials.

    Does not contact the server; see src.tracking.ui.check_connection for that.
    """
    import mlflow

    settings = settings or load_settings()
    if settings.is_remote:
        creds = resolve_credentials(dvc_config)
        if creds is None:
            raise TrackingError(
                "No DagsHub credentials found. Either configure DVC as explained in "
                "README -> Data (dvc remote modify origin --local user/password), or set "
                "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD."
            )
        user, token, _ = creds
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(settings.uri)
    return settings
