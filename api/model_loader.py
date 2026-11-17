"""Choose and load the model the service uses (US6.1 task 2, issue #26).

Which model to use is read from environment variables, so a different model is
picked up just by restarting the service, without changing any code:

- MODEL_SOURCE=baseline (default): the team's popularity baseline
  (src/models/baseline.py), computed from the ratings at startup.
- MODEL_SOURCE=mlflow: the model registered in MLflow as MODEL_NAME with the
  alias MODEL_ALIAS (default "recommender" and "champion"). MLflow is only
  imported in this case, so it is not needed to run the baseline or the tests.

Whatever the source, the result is the same: a table of movies with a score,
ordered from the most to the least recommended.
"""

import os
from dataclasses import dataclass, field

import pandas as pd

from src.models.baseline import compute_movie_stats

DEFAULT_SOURCE = "baseline"
DEFAULT_MODEL_NAME = "recommender"
DEFAULT_MODEL_ALIAS = "champion"


class ModelLoadError(Exception):
    """The model could not be loaded (wrong settings or registry not reachable)."""


@dataclass
class LoadedModel:
    """A model ready to use: its ranked movies and a description for /health."""

    ranked_movies: pd.DataFrame  # movie_id, title, genres, score (best first)
    info: dict = field(default_factory=dict)  # source, name, alias, version

    @property
    def label(self) -> str:
        """Short name sent back in every answer as "source"."""
        if self.info.get("source") == "baseline":
            return "popularity-baseline"
        return f"{self.info.get('name')}@{self.info.get('alias')} v{self.info.get('version')}"


def _attach_titles(scores: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Add title and genres to a (movie_id, score) table and sort it, best first."""
    ranked = scores.merge(
        movies_df[["movie_id", "title", "genres"]], on="movie_id", how="left"
    )
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)


def _load_baseline(ratings_df, movies_df, min_ratings) -> LoadedModel:
    ranked = compute_movie_stats(
        ratings_df, min_ratings=min_ratings, movies_df=movies_df
    )
    info = {
        "source": "baseline",
        "name": "popularity-baseline",
        "alias": None,
        "version": "local",
    }
    return LoadedModel(ranked_movies=ranked, info=info)


def _load_from_mlflow(ratings_df, movies_df, min_ratings, name, alias) -> LoadedModel:
    """Load models:/<name>@<alias> from the MLflow registry.

    Expected model contract (to agree with US4.1): a pyfunc model that receives a
    DataFrame with a "movie_id" column and returns one score per row (a list, an
    array, a Series or a DataFrame with a "score" column).
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ModelLoadError(
            "MODEL_SOURCE=mlflow needs the mlflow package: pip install mlflow"
        ) from exc

    try:
        version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
        model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    except Exception as exc:  # registry down, unknown name or alias...
        raise ModelLoadError(
            f"Could not load models:/{name}@{alias} from MLflow: {exc}"
        ) from exc

    # Only score movies with enough ratings, as the baseline does
    counts = ratings_df.groupby("movie_id").size()
    candidates = pd.DataFrame(
        {"movie_id": counts[counts >= min_ratings].index.astype(int)}
    )
    prediction = model.predict(candidates)
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction["score"]
    candidates["score"] = pd.Series(
        list(prediction), index=candidates.index, dtype=float
    )

    info = {"source": "mlflow", "name": name, "alias": alias, "version": str(version)}
    return LoadedModel(ranked_movies=_attach_titles(candidates, movies_df), info=info)


def load_model(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, min_ratings: int
) -> LoadedModel:
    """Load the model chosen with the environment variables (see the module docstring)."""
    source = os.getenv("MODEL_SOURCE", DEFAULT_SOURCE).strip().lower()
    if source == "baseline":
        return _load_baseline(ratings_df, movies_df, min_ratings)
    if source == "mlflow":
        name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        alias = os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        return _load_from_mlflow(ratings_df, movies_df, min_ratings, name, alias)
    raise ModelLoadError(
        f"Unknown MODEL_SOURCE '{source}'. Use 'baseline' or 'mlflow'."
    )
