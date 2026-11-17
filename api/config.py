"""Settings of the recommendation service, read from params.yaml (issues #28 and #30).

The values live in the `serving` section of params.yaml, at the root of the
repository, so they can be changed without touching the code. An environment
variable with the same name in capitals (for example MIN_RATINGS) overrides the
file, which is handy for a quick test. PARAMS_PATH points to another file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = "params.yaml"


class ConfigError(Exception):
    """params.yaml is missing a value or has a wrong one."""


@dataclass(frozen=True)
class ServingSettings:
    min_ratings: int
    moods: dict[str, list[str]] = field(default_factory=dict)  # mood -> genres


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
    return ServingSettings(min_ratings=min_ratings, moods=_read_moods(serving, path))


def _read_moods(serving: dict, path: Path) -> dict[str, list[str]]:
    """Check the `moods` section: each mood needs a non-empty list of genre names."""
    moods = serving.get("moods") or {}
    if not isinstance(moods, dict):
        raise ConfigError(
            f"'serving.moods' in {path} must map each mood to a list of genres."
        )
    checked = {}
    for mood, genres in moods.items():
        if (
            not isinstance(genres, list)
            or not genres
            or not all(isinstance(g, str) for g in genres)
        ):
            raise ConfigError(f"Mood '{mood}' in {path} needs a list of genre names.")
        checked[str(mood)] = genres
    return checked
