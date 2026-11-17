"""Settings of the recommendation service, read from params.yaml (issue #28).

The values live in the `serving` section of params.yaml, at the root of the
repository, so they can be changed without touching the code. An environment
variable with the same name in capitals (for example MIN_RATINGS) overrides the
file, which is handy for a quick test. PARAMS_PATH points to another file.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = "params.yaml"


class ConfigError(Exception):
    """params.yaml is missing a value or has a wrong one."""


@dataclass(frozen=True)
class ServingSettings:
    min_ratings: int


def load_settings(path: str | Path | None = None) -> ServingSettings:
    """Read the `serving` section of params.yaml (environment variables win)."""
    path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    if not path.exists():
        raise ConfigError(
            f"Settings file not found: {path}. Run the service from the repository root."
        )

    with open(path, encoding="utf-8") as f:
        serving = (yaml.safe_load(f) or {}).get("serving") or {}

    raw = os.getenv("MIN_RATINGS", serving.get("min_ratings"))
    if raw is None:
        raise ConfigError(f"Missing 'serving.min_ratings' in {path}.")
    try:
        min_ratings = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"'min_ratings' must be a whole number, got {raw!r}."
        ) from exc
    if min_ratings < 1:
        raise ConfigError(f"'min_ratings' must be 1 or more, got {min_ratings}.")
    return ServingSettings(min_ratings=min_ratings)
