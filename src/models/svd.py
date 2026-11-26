"""Matrix Factorization / FunkSVD model for collaborative filtering (US 2.2 AC2 & AC6).

Fulfills the US 2.2 Acceptance Criteria:
- AC2: A more advanced model is trained. Given training data, an advanced model
       (matrix factorization / collaborative filtering) produces recommendations for every user.
- AC6: Hyperparameters loaded from configuration file (params.yaml).
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
import yaml

from src.models.reference import BaseRecommenderModel


class MatrixFactorizationSVDModel(BaseRecommenderModel):
    """Collaborative Filtering model using FunkSVD Matrix Factorization with biases.

    Decomposes the sparse user-movie interaction matrix into low-rank latent representations:
        r_hat(u, i) = mu + b_u + b_i + P_u . Q_i

    Parameters are optimized using regularized Stochastic Gradient Descent (SGD) minimizing:
        L = sum (r_ui - r_hat_ui)^2 + lambda * (||P_u||^2 + ||Q_i||^2 + b_u^2 + b_i^2)
    """

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.005,
        reg: float = 0.02,
        epochs: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.reg = float(reg)
        self.epochs = int(epochs)
        self.random_state = int(random_state)

        # Fitted attributes
        self.global_mean: float = 3.0
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.movie_to_idx: Dict[int, int] = {}
        self.idx_to_movie: Dict[int, int] = {}
        self.user_rated_movies: Dict[int, Set[int]] = {}

        self.user_biases: Optional[np.ndarray] = None
        self.movie_biases: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.movie_factors: Optional[np.ndarray] = None

        self.training_history: List[float] = []

    @classmethod
    def from_config(cls, params_path: Union[str, Path] = "params.yaml") -> "MatrixFactorizationSVDModel":
        """Initialize MatrixFactorizationSVDModel using hyperparameters from params.yaml."""
        path = Path(params_path)
        params = {
            "n_factors": 20,
            "learning_rate": 0.005,
            "reg": 0.02,
            "epochs": 15,
            "random_state": 42,
        }
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            svd_cfg = cfg.get("models", {}).get("svd", {})
            params.update({k: v for k, v in svd_cfg.items() if k in params})
        return cls(**params)

    def fit(self, train_df: pd.DataFrame) -> "MatrixFactorizationSVDModel":
        """Fit latent factor vectors and biases on interaction data via SGD."""
        if train_df is None or train_df.empty:
            raise ValueError("train_df cannot be empty or None")

        required_cols = {"user_id", "movie_id", "rating"}
        if not required_cols.issubset(train_df.columns):
            missing = required_cols - set(train_df.columns)
            raise ValueError(f"train_df missing required columns: {missing}")

        self.global_mean = float(train_df["rating"].mean())

        # Build user and movie index mappings
        unique_users = train_df["user_id"].unique()
        unique_movies = train_df["movie_id"].unique()

        self.user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: int(uid) for idx, uid in enumerate(unique_users)}
        self.movie_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie = {idx: int(mid) for idx, mid in enumerate(unique_movies)}

        # Map movies seen by each user for recommendation filtering
        self.user_rated_movies = {
            int(u): set(map(int, movies))
            for u, movies in train_df.groupby("user_id")["movie_id"]
        }

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        self.user_biases = np.zeros(n_users, dtype=np.float64)
        self.movie_biases = np.zeros(n_movies, dtype=np.float64)
        self.user_factors = rng.normal(0.0, 0.1, (n_users, self.n_factors)).astype(np.float64)
        self.movie_factors = rng.normal(0.0, 0.1, (n_movies, self.n_factors)).astype(np.float64)

        u_arr = train_df["user_id"].map(self.user_to_idx).to_numpy(dtype=np.int32)
        m_arr = train_df["movie_id"].map(self.movie_to_idx).to_numpy(dtype=np.int32)
        r_arr = train_df["rating"].to_numpy(dtype=np.float64)

        n_samples = len(r_arr)
        indices = np.arange(n_samples)

        lr = self.learning_rate
        reg = self.reg
        mu = self.global_mean
        P = self.user_factors
        Q = self.movie_factors
        bu = self.user_biases
        bi = self.movie_biases

        self.training_history = []

        for _ in range(self.epochs):
            rng.shuffle(indices)
            total_sq_error = 0.0

            for idx in indices:
                u = u_arr[idx]
                i = m_arr[idx]
                r = r_arr[idx]

                pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
                err = r - pred
                total_sq_error += err * err

                # SGD updates for biases
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

                # SGD updates for latent factors
                p_u_old = P[u].copy()
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * p_u_old - reg * Q[i])

            rmse_epoch = np.sqrt(total_sq_error / n_samples)
            self.training_history.append(float(rmse_epoch))

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating score with fallback for cold-start users or items."""
        if self.user_factors is None or self.movie_factors is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        u_idx = self.user_to_idx.get(int(user_id))
        m_idx = self.movie_to_idx.get(int(movie_id))

        if u_idx is not None and m_idx is not None:
            raw_score = (
                self.global_mean
                + self.user_biases[u_idx]
                + self.movie_biases[m_idx]
                + float(np.dot(self.user_factors[u_idx], self.movie_factors[m_idx]))
            )
        elif u_idx is not None:
            raw_score = self.global_mean + self.user_biases[u_idx]
        elif m_idx is not None:
            raw_score = self.global_mean + self.movie_biases[m_idx]
        else:
            raw_score = self.global_mean

        return float(np.clip(raw_score, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generate batch rating predictions."""
        if test_df is None or test_df.empty:
            return np.array([])
        if self.user_factors is None or self.movie_factors is None:
            raise RuntimeError("Model must be fitted before calling predict_batch()")

        u_ids = test_df["user_id"].to_numpy()
        m_ids = test_df["movie_id"].to_numpy()

        preds = np.empty(len(test_df), dtype=np.float64)
        for idx, (uid, mid) in enumerate(zip(u_ids, m_ids)):
            preds[idx] = self.predict(int(uid), int(mid))
        return preds

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Produce top-N recommendations for a specific user (AC2).

        Returns DataFrame with columns: ['rank', 'movie_id', 'score'].
        """
        if self.movie_factors is None or len(self.movie_to_idx) == 0:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])

        all_movie_ids = np.array(list(self.movie_to_idx.keys()), dtype=int)
        all_movie_indices = np.array(list(self.movie_to_idx.values()), dtype=int)

        seen_movies = self.user_rated_movies.get(int(user_id), set()) if exclude_seen else set()
        if seen_movies:
            mask = np.isin(all_movie_ids, list(seen_movies), invert=True)
            candidate_ids = all_movie_ids[mask]
            candidate_indices = all_movie_indices[mask]
        else:
            candidate_ids = all_movie_ids
            candidate_indices = all_movie_indices

        if len(candidate_ids) == 0:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])

        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is not None:
            u_factor = self.user_factors[u_idx]
            u_bias = self.user_biases[u_idx]
            cand_biases = self.movie_biases[candidate_indices]
            cand_factors = self.movie_factors[candidate_indices]

            scores = self.global_mean + u_bias + cand_biases + np.dot(cand_factors, u_factor)
        else:
            scores = self.global_mean + self.movie_biases[candidate_indices]

        clipped_scores = np.clip(scores, 1.0, 5.0)

        n_take = min(n, len(candidate_ids))
        if n_take < len(candidate_ids):
            top_part = np.argpartition(-clipped_scores, n_take)[:n_take]
            top_sorted = top_part[np.argsort(-clipped_scores[top_part])]
        else:
            top_sorted = np.argsort(-clipped_scores)

        recs_df = pd.DataFrame({
            "rank": np.arange(1, n_take + 1),
            "movie_id": candidate_ids[top_sorted],
            "score": np.round(clipped_scores[top_sorted], 4),
        })
        return recs_df
