"""Log one small training to check that tracking works end to end (US3.2, task 2).

From the repository root, after `dvc pull`:

    python -m src.tracking.check_run

It trains the popularity baseline of src/models/baseline.py on all the ratings and
records it with log_training(): parameters, seed, a few descriptive numbers of the
training and the model. It prints the run id and the address of the run.

This is a *check* of the tracking, not an evaluation: the evaluation scores
(Precision@10, NDCG@10...) of the baseline and the SVD come from the US2.2 training,
which calls the same log_training(). The run is tagged purpose=tracking-check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.models.baseline import compute_movie_stats
from src.tracking.config import TrackingError, load_settings
from src.tracking.runs import log_training, set_seed

DEFAULT_RATINGS = "data/raw/ml-1m/ratings.dat"


class PopularityBaseline:
    """The team's popularity baseline: the same score for every user.

    predict(user, movie) is the weighted rating of the movie, or 0.0 if the movie has
    fewer than `min_ratings` ratings (it is never recommended).
    """

    def __init__(self, min_ratings: int = 50):
        self.min_ratings = int(min_ratings)
        self.scores: dict[int, float] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityBaseline":
        ranked = compute_movie_stats(ratings, min_ratings=self.min_ratings)
        self.scores = dict(zip(ranked["movie_id"].astype(int), ranked["score"].astype(float)))
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.scores.get(int(movie_id), 0.0)


def load_ratings(path: str | Path) -> pd.DataFrame:
    """MovieLens ratings: .dat ('::', no header) or .csv / .parquet with user_id, movie_id, rating."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ratings not found: {path}. Run 'dvc pull' first (README -> Data).")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".dat":
        return pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.tracking.check_run",
        description="Log the popularity baseline to the team's MLflow server to check tracking works.",
    )
    parser.add_argument("--ratings", default=DEFAULT_RATINGS, help="ratings file (default: %(default)s)")
    parser.add_argument("--min-ratings", type=int, default=50, help="minimum ratings per movie")
    parser.add_argument("--seed", type=int, default=42, help="random seed (logged with the run)")
    args = parser.parse_args(argv)

    try:
        ratings = load_ratings(args.ratings)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    seed = set_seed(args.seed)
    model = PopularityBaseline(min_ratings=args.min_ratings).fit(ratings)
    scores = pd.Series(model.scores, dtype=float)

    try:
        settings = load_settings()
        run_id = log_training(
            model,
            model_name="popularity-baseline",
            params={"min_ratings": args.min_ratings, "data": {"ratings": args.ratings}},
            seed=seed,
            metrics={
                "train_ratings": len(ratings),
                "ranked_movies": len(scores),
                "mean_score": float(scores.mean()) if len(scores) else 0.0,
                "top_score": float(scores.max()) if len(scores) else 0.0,
            },
            example=ratings[["user_id", "movie_id"]].head(5),
            tags={"purpose": "tracking-check"},
            data_paths=[args.ratings],
            settings=settings,
        )
    except TrackingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    import mlflow

    experiment_id = mlflow.get_run(run_id).info.experiment_id
    print(f"Run id : {run_id}")
    print(f"Open   : {settings.uri}/#/experiments/{experiment_id}/runs/{run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
