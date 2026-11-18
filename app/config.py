"""Settings for the recommendation web application (US 7.2).

Reads the service address from the `app` section of params.yaml, or from the
SERVICE_URL environment variable if set.
"""

import os
from pathlib import Path
import yaml

DEFAULT_PARAMS_PATH = "params.yaml"
DEFAULT_SERVICE_URL = "http://127.0.0.1:8000"


def get_service_url(path: str | Path | None = None) -> str:
    """Return the base URL of the recommendation service."""
    # Environment variable has top priority
    if env_url := os.getenv("SERVICE_URL"):
        return env_url.rstrip("/")

    target_path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            app_cfg = data.get("app") or {}
            if service_url := app_cfg.get("service_url"):
                return str(service_url).rstrip("/")

    return DEFAULT_SERVICE_URL
