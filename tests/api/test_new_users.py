"""Tests for best-rated suggestions for new users (issue #28)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config import ConfigError, load_settings
from api.main import create_app
from api.service import RecommenderService


@pytest.fixture
def ratings_with_rare_movie(ratings):
    # Movie 99 has a perfect average but only 2 ratings: it must not be suggested
    rare = pd.DataFrame(
        {
            "user_id": [1, 2],
            "movie_id": [99, 99],
            "rating": [5, 5],
            "timestamp": [978300000, 978300001],
        }
    )
    return pd.concat([ratings, rare], ignore_index=True)


@pytest.fixture
def movies_with_rare_movie(movies):
    rare = pd.DataFrame(
        {"movie_id": [99], "title": ["Rare Masterpiece"], "genres": ["Drama"]}
    )
    return pd.concat([movies, rare], ignore_index=True)


def test_new_user_gets_at_least_5_best_rated_movies_in_order(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={})
    assert response.status_code == 200
    movies = response.json()["recommendations"]
    assert len(movies) >= 5
    scores = [m["score"] for m in movies]
    assert scores == sorted(scores, reverse=True)


def test_movies_below_the_minimum_are_not_suggested(
    ratings_with_rare_movie, movies_with_rare_movie
):
    service = RecommenderService(
        ratings_with_rare_movie, movies_with_rare_movie, min_ratings=3
    )
    with TestClient(create_app(service)) as client:
        movies = client.post("/recommend", json={"n": 50}).json()["recommendations"]
    assert 99 not in {m["movie_id"] for m in movies}


def test_the_minimum_changes_the_suggestions(
    ratings_with_rare_movie, movies_with_rare_movie
):
    service = RecommenderService(
        ratings_with_rare_movie, movies_with_rare_movie, min_ratings=1
    )
    with TestClient(create_app(service)) as client:
        movies = client.post("/recommend", json={"n": 50}).json()["recommendations"]
    assert 99 in {m["movie_id"] for m in movies}


def test_unknown_user_still_gets_the_us61_error(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"user_id": 99999})
    assert response.status_code == 404


def test_minimum_is_read_from_params_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    params = tmp_path / "params.yaml"
    params.write_text("serving:\n  min_ratings: 25\n")
    assert load_settings(params).min_ratings == 25


def test_environment_variable_overrides_the_file(tmp_path, monkeypatch):
    params = tmp_path / "params.yaml"
    params.write_text("serving:\n  min_ratings: 25\n")
    monkeypatch.setenv("MIN_RATINGS", "10")
    assert load_settings(params).min_ratings == 10


def test_project_params_yaml_has_a_valid_minimum(monkeypatch):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    assert load_settings("params.yaml").min_ratings >= 1


@pytest.mark.parametrize(
    "content",
    [
        "serving: {}\n",
        "serving:\n  min_ratings: zero\n",
        "serving:\n  min_ratings: 0\n",
    ],
)
def test_wrong_settings_give_a_clear_error(tmp_path, monkeypatch, content):
    monkeypatch.delenv("MIN_RATINGS", raising=False)
    params = tmp_path / "params.yaml"
    params.write_text(content)
    with pytest.raises(ConfigError):
        load_settings(params)


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")
