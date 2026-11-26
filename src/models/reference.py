"""Simple reference baseline models for recommender systems benchmark.

Fulfills the US 2.2 Acceptance Criteria:
- AC1: Simple reference models exist. Given cleaned training data, basic models produce
       recommendations used as a benchmark starting point.
- AC4: Evaluated using standard prediction and ranking interfaces.
- AC6: Hyperparameters loaded from configuration file.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

from src.models.baseline import compute_movie_stats, get_user_recommendations


class BaseRecommenderModel(ABC):
    """Abstract base class for recommendation models."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommenderModel":
        """Fit the model on training ratings data."""
        pass

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating score for a specific user and movie."""
        pass

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate rating predictions for a batch of interactions."""
        if test_df is None or test_df.empty:
            return np.array([])
        predictions = [
            self.predict(int(row["user_id"]), int(row["movie_id"]))
            for _, row in test_df.iterrows()
        ]
        return np.array(predictions)

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Produce top-N recommendations for a specific user."""
        pass


class GlobalMeanModel(BaseRecommenderModel):
    """Naive reference baseline that predicts the global average rating for all pairs."""

    def __init__(self) -> None:
        self.global_mean: float = 3.0
        self.movie_means: Dict[int, float] = {}
        self.train_df: Optional[pd.DataFrame] = None
        self.candidate_movies: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> "GlobalMeanModel":
        """Fit by computing the global mean and per-movie average ratings."""
        if train_df is None or train_df.empty:
            raise ValueError("train_df cannot be empty or None")

        self.train_df = train_df.copy()
        self.global_mean = float(train_df["rating"].mean())

        # Movie-level averages for ranking
        stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
        stats.columns = ["movie_id", "rating_count", "score"]
        self.candidate_movies = stats.sort_values(by="score", ascending=False).reset_index(drop=True)
        self.movie_means = dict(zip(stats["movie_id"], stats["score"]))
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating as global mean (constant baseline)."""
        return self.global_mean

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Generate top-N recommendations based on overall movie averages."""
        if self.candidate_movies is None or self.candidate_movies.empty:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])

        return get_user_recommendations(
            user_id=user_id,
            ranked_movies_df=self.candidate_movies,
            ratings_df=self.train_df if exclude_seen else None,
            n=n,
            exclude_rated=exclude_seen,
        )


class PopularityReferenceModel(BaseRecommenderModel):
    """Weighted Bayesian popularity baseline (IMDb formula)."""

    def __init__(self, min_ratings: int = 50) -> None:
        self.min_ratings = min_ratings
        self.global_mean: float = 3.0
        self.ranked_movies: Optional[pd.DataFrame] = None
        self.movie_scores: Dict[int, float] = {}
        self.train_df: Optional[pd.DataFrame] = None

    @classmethod
    def from_config(cls, params_path: Union[str, Path] = "params.yaml") -> "PopularityReferenceModel":
        """Initialize PopularityReferenceModel using settings from params.yaml."""
        path = Path(params_path)
        min_ratings = 50
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            min_ratings = int(
                cfg.get("models", {}).get("baselines", {}).get("popularity", {}).get("min_ratings", 50)
            )
        return cls(min_ratings=min_ratings)

    def fit(
        self,
        train_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> "PopularityReferenceModel":
        """Fit popularity model by computing Bayesian scores on the training data."""
        if train_df is None or train_df.empty:
            raise ValueError("train_df cannot be empty or None")

        self.train_df = train_df.copy()
        self.global_mean = float(train_df["rating"].mean())

        self.ranked_movies = compute_movie_stats(
            ratings_df=train_df,
            min_ratings=self.min_ratings,
            movies_df=movies_df,
        )

        if not self.ranked_movies.empty and "score" in self.ranked_movies.columns:
            self.movie_scores = dict(zip(self.ranked_movies["movie_id"], self.ranked_movies["score"]))
        else:
            self.movie_scores = {}

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating as the movie's weighted score, falling back to global mean."""
        return float(self.movie_scores.get(movie_id, self.global_mean))

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Generate top-N recommendations using the weighted Bayesian baseline."""
        if self.ranked_movies is None or self.ranked_movies.empty:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])

        return get_user_recommendations(
            user_id=user_id,
            ranked_movies_df=self.ranked_movies,
            ratings_df=self.train_df if exclude_seen else None,
            n=n,
            exclude_rated=exclude_seen,
        )
