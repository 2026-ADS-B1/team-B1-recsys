"""Tests for recommendations by mood (issue #30)."""

import pytest
from fastapi.testclient import TestClient

from api.config import ConfigError, load_settings
from api.main import create_app
from api.service import InvalidMoodConfigError, RecommenderService

MOODS = {"relaxed": ["Comedy", "Animation"], "romantic": ["Romance"]}


@pytest.fixture
def mood_service(ratings, movies):
    return RecommenderService(ratings, movies, min_ratings=3, moods=MOODS)


def test_a_mood_returns_movies_from_its_genres(mood_service):
    with TestClient(create_app(mood_service)) as client:
        response = client.post("/recommend", json={"mood": "relaxed"})
    assert response.status_code == 200
    body = response.json()
    assert body["mood"] == "relaxed"
    assert len(body["recommendations"]) >= 5
    assert all(
        {"Comedy", "Animation"} & set(m["genres"]) for m in body["recommendations"]
    )
    scores = [m["score"] for m in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_known_user_with_a_mood_gets_no_rated_movies(mood_service, ratings):
    rated = set(ratings.loc[ratings.user_id == 1, "movie_id"])
    with TestClient(create_app(mood_service)) as client:
        movies = client.post(
            "/recommend", json={"user_id": 1, "mood": "Relaxed"}
        ).json()["recommendations"]
    assert movies
    assert not {m["movie_id"] for m in movies} & rated


def test_unknown_mood_gets_a_clear_error(mood_service):
    with TestClient(create_app(mood_service)) as client:
        response = client.post("/recommend", json={"mood": "hungry"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "hungry" in detail and "relaxed" in detail


def test_genre_and_mood_together_get_a_clear_error(mood_service):
    with TestClient(create_app(mood_service)) as client:
        response = client.post("/recommend", json={"genre": "Drama", "mood": "relaxed"})
    assert response.status_code == 422
    assert "not both" in response.json()["detail"]


def test_moods_endpoint_lists_moods_and_genres(mood_service):
    with TestClient(create_app(mood_service)) as client:
        moods = client.get("/moods").json()["moods"]
    assert moods == {"relaxed": ["Comedy", "Animation"], "romantic": ["Romance"]}


def test_a_mood_with_an_unknown_genre_is_rejected_at_startup(ratings, movies):
    with pytest.raises(InvalidMoodConfigError, match="Telenovela"):
        RecommenderService(
            ratings, movies, min_ratings=3, moods={"weird": ["Telenovela"]}
        )


def test_moods_are_read_from_params_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    params = tmp_path / "params.yaml"
    params.write_text("serving:\n  min_ratings: 5\n  moods:\n    relaxed: [Comedy]\n")
    assert load_settings(params).moods == {"relaxed": ["Comedy"]}


def test_a_badly_written_mood_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    params = tmp_path / "params.yaml"
    params.write_text("serving:\n  min_ratings: 5\n  moods:\n    relaxed: Comedy\n")
    with pytest.raises(ConfigError, match="relaxed"):
        load_settings(params)


def test_project_moods_only_use_movielens_genres(monkeypatch):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    movielens = {
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    }
    moods = load_settings("params.yaml").moods
    assert moods, "params.yaml should define at least one mood"
    for genres in moods.values():
        assert set(genres) <= movielens
