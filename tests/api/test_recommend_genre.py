from fastapi.testclient import TestClient

from api.main import create_app


def test_new_user_with_genre_gets_only_that_genre(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"genre": "Comedy"})
    assert response.status_code == 200
    movies = response.json()["recommendations"]
    assert len(movies) == 5
    assert all("Comedy" in m["genres"] for m in movies)
    scores = [m["score"] for m in movies]
    assert scores == sorted(scores, reverse=True)


def test_known_user_with_genre_gets_that_genre_without_rated_movies(service, ratings):
    rated = set(ratings.loc[ratings.user_id == 1, "movie_id"])
    with TestClient(create_app(service)) as client:
        movies = client.post(
            "/recommend", json={"user_id": 1, "genre": "Drama"}
        ).json()["recommendations"]
    assert movies
    assert all("Drama" in m["genres"] for m in movies)
    assert not {m["movie_id"] for m in movies} & rated


def test_genre_is_case_insensitive(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"genre": "comedy"})
    assert response.status_code == 200
    assert response.json()["genre"] == "Comedy"


def test_unknown_genre_gets_a_clear_error(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/recommend", json={"genre": "Telenovela"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Telenovela" in detail and "Comedy" in detail


def test_new_user_without_genre_gets_the_best_rated(service):
    with TestClient(create_app(service)) as client:
        movies = client.post("/recommend", json={}).json()["recommendations"]
    assert len(movies) == 5


def test_genres_endpoint_lists_the_genres(service):
    with TestClient(create_app(service)) as client:
        genres = client.get("/genres").json()["genres"]
    assert "Comedy" in genres and "Drama" in genres
