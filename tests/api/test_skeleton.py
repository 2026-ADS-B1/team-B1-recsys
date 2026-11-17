from fastapi.testclient import TestClient

from api.main import create_app


def test_docs_page_opens(service):
    with TestClient(create_app(service)) as client:
        assert client.get("/docs").status_code == 200


def test_examples_are_in_the_documentation(service):
    with TestClient(create_app(service)) as client:
        schema = client.get("/openapi.json").json()
    body = schema["paths"]["/recommend"]["post"]["requestBody"]["content"][
        "application/json"
    ]
    assert "known_user" in body["examples"]
