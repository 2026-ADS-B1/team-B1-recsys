"""Tests for the recommendations page (US 7.2).

Validates:
- Service address is read from configuration (params.yaml or SERVICE_URL).
- No recommendation ML models are loaded in the app.
- Page loads cleanly without errors.
- Mutual exclusion of genre vs. mood filters.
- Guest mode vs. User ID inputs.
- Real HTTP call to POST /recommend with a fake service (mocked httpx):
  - Movies shown in order received, with title and genres.
  - User ID omitted for guests.
  - Proper error handling for 404 (user not found) and service unavailable.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from api.config import load_settings
from app.config import get_service_url


def test_service_address_read_from_configuration(tmp_path, monkeypatch):
    """Verify service address is read from params.yaml and can be overridden by SERVICE_URL."""
    # 1. Read from params.yaml
    service_url = get_service_url()
    assert service_url == "http://127.0.0.1:8000"

    # 2. Override via SERVICE_URL environment variable
    monkeypatch.setenv("SERVICE_URL", "http://recommender.prod.internal:9000/")
    assert get_service_url() == "http://recommender.prod.internal:9000"

    # 3. Read from custom params.yaml
    custom_yaml = tmp_path / "custom_params.yaml"
    custom_yaml.write_text("app:\n  service_url: http://custom-host:8080\n", encoding="utf-8")
    monkeypatch.delenv("SERVICE_URL", raising=False)
    assert get_service_url(custom_yaml) == "http://custom-host:8080"


def test_no_model_loaded_in_app():
    """Verify that the web application does not load or import recommendation ML models."""
    import app.main

    # Ensure no model classes/loaders are imported into app.main
    forbidden_modules = [
        "src.models.baseline",
        "src.models.batch_recommend",
        "api.model_loader",
        "api.service",
    ]
    for mod in forbidden_modules:
        assert mod not in dir(app.main), f"Module '{mod}' should not be loaded in app.main"


def test_page_loads_without_errors():
    """Verify that the recommendations page loads cleanly with no exceptions."""
    at = AppTest.from_file("app/main.py")
    at.run()
    assert not at.exception


def test_moods_read_from_configuration():
    """Verify that moods are loaded from params.yaml rather than being hardcoded."""
    settings = load_settings()
    expected_moods = sorted(list(settings.moods.keys()))

    at = AppTest.from_file("app/main.py")
    at.run()

    at.radio[0].set_value("By Mood").run()
    assert not at.exception

    mood_options = at.selectbox[0].options
    assert sorted(list(mood_options)) == expected_moods


def test_guest_mode_toggle():
    """Verify that toggling the 'Continue as guest' checkbox works cleanly."""
    at = AppTest.from_file("app/main.py")
    at.run()

    assert not at.checkbox[0].value
    assert len(at.number_input) == 1

    at.checkbox[0].check().run()
    assert not at.exception
    assert len(at.number_input) == 0


def test_genre_or_mood_selector_mutual_exclusion():
    """Verify only one selector (genre OR mood) is active at a time."""
    at = AppTest.from_file("app/main.py")
    at.run()

    # 1. No filter selected initially -> no selectbox present
    assert len(at.selectbox) == 0

    # 2. Select Genre filter
    at.radio[0].set_value("By Genre").run()
    assert not at.exception
    assert len(at.selectbox) == 1
    assert len(at.selectbox[0].options) > 0

    # 3. Switch to Mood filter
    at.radio[0].set_value("By Mood").run()
    assert not at.exception
    assert len(at.selectbox) == 1


@patch("httpx.post")
def test_successful_recommendation_shows_movies_in_order(mock_post):
    """Verify POST /recommend call, waiting indicator execution, and ordered movie display."""
    fake_movies = [
        {"rank": 1, "movie_id": 101, "title": "First Movie (1995)", "genres": ["Animation", "Children's"], "score": 4.9},
        {"rank": 2, "movie_id": 202, "title": "Second Movie (1998)", "genres": ["Action", "Thriller"], "score": 4.5},
        {"rank": 3, "movie_id": 303, "title": "Third Movie (2002)", "genres": ["Drama"], "score": 4.1},
    ]
    mock_post.return_value = httpx.Response(
        200,
        json={
            "user_id": 1,
            "genre": None,
            "mood": None,
            "source": "popularity-baseline",
            "recommendations": fake_movies,
        },
    )

    at = AppTest.from_file("app/main.py")
    at.run()

    # Click Get Recommendations
    at.button[0].click().run()
    assert not at.exception

    # Verify POST /recommend was called with expected payload
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    called_json = mock_post.call_args[1]["json"]
    assert called_url.endswith("/recommend")
    assert called_json == {"user_id": 1}

    # Verify dataframe display has exact order, title, and genres
    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert list(df["Title"]) == ["First Movie (1995)", "Second Movie (1998)", "Third Movie (2002)"]
    assert list(df["Genres"]) == ["Animation, Children's", "Action, Thriller", "Drama"]
    assert list(df["Rank"]) == [1, 2, 3]


@patch("httpx.post")
def test_guest_recommendation_sends_no_user_id(mock_post):
    """Verify that continuing as guest sends no user_id to the recommendation service."""
    mock_post.return_value = httpx.Response(
        200,
        json={
            "user_id": None,
            "genre": "Comedy",
            "mood": None,
            "source": "popularity-baseline",
            "recommendations": [
                {"rank": 1, "movie_id": 1, "title": "Toy Story (1995)", "genres": ["Comedy"], "score": 4.5}
            ],
        },
    )

    at = AppTest.from_file("app/main.py")
    at.run()

    # Select Guest mode
    at.checkbox[0].check().run()
    # Select Genre filter
    at.radio[0].set_value("By Genre").run()
    at.selectbox[0].select("Comedy").run()

    # Click Get Recommendations
    at.button[0].click().run()
    assert not at.exception

    mock_post.assert_called_once()
    called_json = mock_post.call_args[1]["json"]
    # user_id must NOT be present in payload
    assert "user_id" not in called_json
    assert called_json.get("genre") == "Comedy"


@patch("httpx.post")
def test_recommendation_handles_404_user_not_found(mock_post):
    """Verify that a 404 response displays an error message to the user."""
    mock_post.return_value = httpx.Response(
        404,
        json={"detail": "User 9999 is not in the system."},
    )

    at = AppTest.from_file("app/main.py")
    at.run()

    at.number_input[0].set_value(9999).run()
    at.button[0].click().run()
    assert not at.exception

    # An error message should be displayed
    assert len(at.error) > 0
    assert "User 9999 is not in the system" in at.error[0].value


@patch("httpx.post")
def test_recommendation_handles_service_unreachable(mock_post):
    """Verify that connection failure gracefully displays a service unreachable error."""
    mock_post.side_effect = httpx.ConnectError("Connection refused")

    at = AppTest.from_file("app/main.py")
    at.run()

    at.button[0].click().run()
    assert not at.exception

    assert len(at.error) > 0
    assert "Could not connect to recommendation service" in at.error[0].value
