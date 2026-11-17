"""Read the MovieLens files (same formats as the batch job: .dat with '::', .csv or .parquet)."""

from pathlib import Path

import pandas as pd


def _read_table(path: str | Path, dat_columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run 'dvc pull' or set the path with an environment variable."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    with open(path, "r", encoding="latin-1") as f:
        first_line = f.readline()
    if "::" in first_line:
        return pd.read_csv(
            path, sep="::", engine="python", names=dat_columns, encoding="latin-1"
        )
    return pd.read_csv(path)


def load_ratings(path: str | Path) -> pd.DataFrame:
    return _read_table(path, ["user_id", "movie_id", "rating", "timestamp"])


def load_movies(path: str | Path) -> pd.DataFrame:
    return _read_table(path, ["movie_id", "title", "genres"])
