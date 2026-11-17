from fastapi.testclient import TestClient

from api.main import create_app


def test_docs_page_opens():
    client = TestClient(create_app())
    assert client.get("/docs").status_code == 200


def test_examples_are_in_the_documentation():
    schema = TestClient(create_app()).get("/openapi.json").json()
    body = schema["paths"]["/recommend"]["post"]["requestBody"]["content"][
        "application/json"
    ]
    assert "known_user" in body["examples"]


def test_recommend_returns_an_ordered_list_with_title_and_genres():
    response = TestClient(create_app()).post("/recommend", json={"user_id": 1})
    assert response.status_code == 200
    movies = response.json()["recommendations"]
    assert len(movies) >= 5
    assert [m["rank"] for m in movies] == list(range(1, len(movies) + 1))
    assert all(m["title"] and m["genres"] for m in movies)
