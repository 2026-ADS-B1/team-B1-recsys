"""Recommendation logic used by the API.

The ranked movies come from the model chosen at startup (api/model_loader.py):
by default the team's popularity baseline (src/models/baseline.py).
"""

import os

import pandas as pd

from api.data import load_movies, load_ratings
from api.model_loader import LoadedModel, load_model

DEFAULT_RATINGS_PATH = "data/raw/ml-1m/ratings.dat"
DEFAULT_MOVIES_PATH = "data/raw/ml-1m/movies.dat"
DEFAULT_MIN_RATINGS = 50


class UserNotFoundError(Exception):
    """The user_id is not in the ratings data."""


class UnknownGenreError(Exception):
    """The genre does not exist in the catalogue."""


class RecommenderService:
    """Loads the data once and answers recommendation requests quickly."""

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        min_ratings: int = DEFAULT_MIN_RATINGS,
        model: LoadedModel | None = None,
    ):
        # 1. The model in use ranks the movies (score from highest to lowest).
        #    Without a model given, use the one chosen in the environment (baseline by default).
        if model is None:
            model = load_model(ratings_df, movies_df, min_ratings)
        self.model_info = model.info
        self.source = model.label
        ranked = model.ranked_movies

        # 2. Keep a light list of the ranked movies, ready to send back
        self._items: list[dict] = [
            {
                "movie_id": int(row.movie_id),
                "title": str(row.title) if pd.notna(row.title) else "Unknown title",
                "genres": str(row.genres).split("|") if pd.notna(row.genres) else [],
                "score": float(row.score),
            }
            for row in ranked.itertuples(index=False)
        ]

        # 3. For every genre, the positions of its movies in the ranked list (case is ignored)
        self._by_genre: dict[str, list[int]] = {}
        for position, item in enumerate(self._items):
            for genre in item["genres"]:
                self._by_genre.setdefault(genre, []).append(position)
        self._genre_lookup = {g.lower(): g for g in self._by_genre}
        self.genres = sorted(self._by_genre)

        # 4. For every user, the set of movies they have already rated
        self._rated: dict[int, frozenset[int]] = {
            int(user): frozenset(movies)
            for user, movies in ratings_df.groupby("user_id")["movie_id"]
        }

    @classmethod
    def from_env(cls) -> "RecommenderService":
        """Build the service from the files given by environment variables (or the defaults)."""
        ratings = load_ratings(os.getenv("RATINGS_PATH", DEFAULT_RATINGS_PATH))
        movies = load_movies(os.getenv("MOVIES_PATH", DEFAULT_MOVIES_PATH))
        min_ratings = int(os.getenv("MIN_RATINGS", DEFAULT_MIN_RATINGS))
        return cls(ratings, movies, min_ratings=min_ratings)

    def canonical_genre(self, genre: str) -> str:
        """'drama' or ' Drama ' -> 'Drama'. Raises UnknownGenreError if it doesn't exist."""
        canonical = self._genre_lookup.get(genre.strip().lower())
        if canonical is None:
            raise UnknownGenreError(genre)
        return canonical

    def recommend(
        self, user_id: int | None = None, genre: str | None = None, n: int = 5
    ) -> list[dict]:
        """Top-n movies, ordered by score.

        - user_id given: the user must exist, and the movies they already rated are left out.
        - user_id missing: a new user, who gets the best-rated movies.
        - genre given: only movies of that genre.
        """
        if user_id is not None and user_id not in self._rated:
            raise UserNotFoundError(user_id)
        rated = (
            self._rated.get(user_id, frozenset())
            if user_id is not None
            else frozenset()
        )

        if genre is not None:
            canonical = self.canonical_genre(genre)
            candidates = (self._items[i] for i in self._by_genre[canonical])
        else:
            candidates = iter(self._items)

        picked = []
        for item in candidates:  # already ordered by score
            if item["movie_id"] in rated:
                continue
            picked.append(item)
            if len(picked) == n:
                break
        return [{"rank": i + 1, **item} for i, item in enumerate(picked)]
