"""Recommendations web page (US 7.2).

Connects to the recommendation API service (POST /recommend).
Start with:
    streamlit run app/main.py
"""

from pathlib import Path
import httpx
import pandas as pd
import streamlit as st

from api.config import load_settings
from api.data import load_movies
from app.config import get_service_url

DEFAULT_MOVIES_PATH = Path("data/raw/ml-1m/movies.dat")


@st.cache_data
def get_moods() -> list[str]:
    """Read available moods dynamically from params.yaml (US6.2 configuration)."""
    settings = load_settings()
    return sorted(list(settings.moods.keys()))


@st.cache_data
def get_genres(service_url: str) -> list[str]:
    """Read available genres from the recommendation service or movie catalogue."""
    # Try fetching from the recommendation service first
    try:
        resp = httpx.get(f"{service_url}/genres", timeout=2.0)
        if resp.status_code == 200:
            genres = resp.json().get("genres", [])
            if genres:
                return sorted(genres)
    except Exception:
        pass

    # Fallback to local movie catalogue
    if DEFAULT_MOVIES_PATH.exists():
        try:
            movies_df = load_movies(DEFAULT_MOVIES_PATH)
            genres_set = set()
            for item in movies_df["genres"].dropna():
                for g in str(item).split("|"):
                    if g.strip():
                        genres_set.add(g.strip())
            return sorted(list(genres_set))
        except Exception:
            pass

    # Default MovieLens 1M catalogue genres
    return [
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
    ]


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

    service_url = get_service_url()

    st.title("🎬 Movie Recommendations")
    st.markdown(
        "Welcome! Get personalised movie recommendations for tonight based on your profile, preferred genre, or mood."
    )

    st.divider()

    # Section 1: User Profile & Guest Option
    st.subheader("1. User Identification")
    is_guest = st.checkbox("Continue as guest", value=False, help="Explore recommendations without logging in.")

    if is_guest:
        user_id = None
        st.info("Browsing as **Guest** (new user without rating history).")
    else:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            step=1,
            value=1,
            help="Enter your MovieLens user ID to get personalised recommendations.",
        )

    st.divider()

    # Section 2: Genre or Mood Selector (only one at a time)
    st.subheader("2. Filter Preferences")
    st.caption("Choose to filter by a genre or by a mood, but not both.")

    filter_type = st.radio(
        "Choose filter mode:",
        options=["No filter", "By Genre", "By Mood"],
        horizontal=True,
    )

    selected_genre = None
    selected_mood = None

    if filter_type == "By Genre":
        genres = get_genres(service_url)
        selected_genre = st.selectbox("Select a genre from the catalogue:", options=genres)
    elif filter_type == "By Mood":
        moods = get_moods()
        selected_mood = st.selectbox(
            "Select a mood (from configuration):",
            options=moods,
        )

    st.divider()

    # Section 3: Recommendations Trigger & Display
    st.subheader("3. Recommendations")

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        payload = {}
        if user_id is not None:
            payload["user_id"] = int(user_id)
        if selected_genre:
            payload["genre"] = selected_genre
        elif selected_mood:
            payload["mood"] = selected_mood

        # Waiting indicator while the service answers
        with st.spinner("Fetching recommendations from service..."):
            try:
                response = httpx.post(
                    f"{service_url}/recommend",
                    json=payload,
                    timeout=5.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    user_display = "Guest" if is_guest else f"User {user_id}"
                    filter_display = (
                        f"Genre: **{selected_genre}**"
                        if selected_genre
                        else (f"Mood: **{selected_mood.capitalize()}**" if selected_mood else "None")
                    )
                    st.subheader("Recommended for You Tonight")
                    st.caption(f"Profile: **{user_display}** | Filter: {filter_display}")

                    if not recommendations:
                        st.info("No recommendations found matching your criteria.")
                    else:
                        # Display movies in the exact order received, with title and genres
                        table_data = []
                        for rec in recommendations:
                            genres_formatted = ", ".join(rec.get("genres", []))
                            table_data.append(
                                {
                                    "Rank": rec.get("rank"),
                                    "Title": rec.get("title"),
                                    "Genres": genres_formatted,
                                    "Score": f"{rec.get('score', 0.0):.2f}",
                                }
                            )

                        df_display = pd.DataFrame(table_data)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)

                elif response.status_code == 404:
                    detail = response.json().get("detail", f"User {user_id} was not found.")
                    st.error(f"User not found: {detail}")
                elif response.status_code == 422:
                    detail = response.json().get("detail", "Invalid request parameters.")
                    st.error(f"Validation error: {detail}")
                else:
                    st.error(f"Recommendation service error ({response.status_code}): {response.text}")

            except (httpx.ConnectError, httpx.TimeoutException):
                st.error(
                    f"Could not connect to recommendation service at {service_url}. "
                    "Please make sure the API service is running (`uvicorn api.main:app`)."
                )
            except Exception as exc:
                st.error(f"Unexpected error when calling recommendation service: {exc}")


if __name__ == "__main__":
    main()
