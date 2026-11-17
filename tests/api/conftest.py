"""Small fake MovieLens data so the API tests run in a second, without the real files."""

import pandas as pd
import pytest

from api.service import RecommenderService


@pytest.fixture
def movies():
    genres = [
        "Comedy",
        "Drama",
        "Action|Comedy",
        "Drama|Romance",
        "Animation|Children's|Comedy",
    ]
    return pd.DataFrame(
        {
            "movie_id": range(1, 21),
            "title": [f"Movie {i}" for i in range(1, 21)],
            "genres": [genres[i % len(genres)] for i in range(20)],
        }
    )


@pytest.fixture
def ratings():
    # 30 users; movie m gets a higher rating the smaller m is, so movie 1 is the most popular
    rows = []
    for user in range(1, 31):
        for movie in range(1, 21):
            if (user + movie) % 3 != 0:
                rows.append(
                    (user, movie, 5 - (movie - 1) // 5, 978300000 + user * 100 + movie)
                )
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


@pytest.fixture
def service(ratings, movies):
    return RecommenderService(ratings, movies, min_ratings=3)
