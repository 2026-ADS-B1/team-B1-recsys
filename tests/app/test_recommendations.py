"""Tests for the recommendations page (US 7.2).

Validates:
- Page loads without errors.
- Guest option and User ID interaction.
- Genre or Mood selector mutual exclusion.
- Moods are read from configuration (params.yaml), not hardcoded.
- Recommendations button triggers example results display.
"""

from streamlit.testing.v1 import AppTest
from api.config import load_settings


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

    # Switch to "By Mood" filter
    at.radio[0].set_value("By Mood").run()
    assert not at.exception

    # The selectbox options must match exactly the moods from params.yaml
    mood_options = at.selectbox[0].options
    assert sorted(list(mood_options)) == expected_moods


def test_guest_mode_toggle():
    """Verify that toggling the 'Continue as guest' checkbox works cleanly."""
    at = AppTest.from_file("app/main.py")
    at.run()

    # Initially, guest mode is unchecked and user_id input exists
    assert not at.checkbox[0].value
    assert len(at.number_input) == 1

    # Check "Continue as guest"
    at.checkbox[0].check().run()
    assert not at.exception
    # Number input is replaced by guest info
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
    # Verify genres are loaded
    assert len(at.selectbox[0].options) > 0

    # 3. Switch to Mood filter
    at.radio[0].set_value("By Mood").run()
    assert not at.exception
    # Still exactly one selectbox active, never both
    assert len(at.selectbox) == 1


def test_get_recommendations_button_shows_example_results():
    """Verify that clicking 'Get Recommendations' displays the example results dataframe."""
    at = AppTest.from_file("app/main.py")
    at.run()

    # Click the recommendation button
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception

    # Confirm example dataframe is rendered
    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert not df.empty
    assert "Rank" in df.columns
    assert "Title" in df.columns
    assert "Genres" in df.columns
    assert "Score" in df.columns
