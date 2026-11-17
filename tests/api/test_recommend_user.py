import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.service import RecommenderService


def test_known_user_gets_5_ordered_movies_with_title_and_genres(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"user_id": 1})
    assert response.status_code == 200
    movies = response.json()["recommendations"]
    assert len(movies) == 5
    scores = [m["score"] for m in movies]
    assert scores == sorted(scores, reverse=True)
    assert [m["rank"] for m in movies] == [1, 2, 3, 4, 5]
    assert all(m["title"] and m["genres"] for m in movies)


def test_movies_already_rated_are_left_out(service, ratings):
    rated = set(ratings.loc[ratings.user_id == 1, "movie_id"])
    with TestClient(create_app(service)) as client:
        movies = client.post("/recommend", json={"user_id": 1, "n": 10}).json()[
            "recommendations"
        ]
    assert movies, "expected some recommendations"
    assert not {m["movie_id"] for m in movies} & rated


def test_unknown_user_gets_a_clear_error(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"user_id": 99999})
    assert response.status_code == 404
    assert "99999" in response.json()["detail"]


def test_answer_arrives_in_less_than_one_second():
    # Data of a realistic size: 3,000 movies, 6,000 users, 300,000 ratings
    rng = np.random.default_rng(0)
    ratings = pd.DataFrame(
        {
            "user_id": rng.integers(1, 6001, 300_000),
            "movie_id": rng.integers(1, 3001, 300_000),
            "rating": rng.integers(1, 6, 300_000),
            "timestamp": 978300000,
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": range(1, 3001),
            "title": [f"M{i}" for i in range(1, 3001)],
            "genres": "Drama|Comedy",
        }
    )
    with TestClient(create_app(RecommenderService(ratings, movies))) as client:
        start = time.perf_counter()
        response = client.post("/recommend", json={"user_id": 42})
        elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 1.0
