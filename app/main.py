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

    if "is_guest" not in st.session_state:
        st.session_state["is_guest"] = False

    def _sync_guest():
        st.session_state["is_guest"] = st.session_state.get("guest_checkbox", False)

    is_guest = st.checkbox(
        "Continue as guest",
        value=st.session_state["is_guest"],
        key="guest_checkbox",
        on_change=_sync_guest,
        help="Explore recommendations without logging in.",
    )
    st.session_state["is_guest"] = is_guest

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
                    # Friendly message when user ID is not in the system, offering guest mode (AC4)
                    st.warning(
                        f"We couldn't find User ID **{user_id}** in our system. "
                        "Would you like to continue as a guest to get our top recommendations for tonight?"
                    )
                    if st.button("👉 Continue as guest", key="btn_continue_as_guest"):
                        st.session_state["guest_checkbox"] = True
                        st.session_state["is_guest"] = True
                        st.rerun()

                elif response.status_code == 422:
                    st.error("Invalid request parameters. Please verify your selected genre or mood.")

                else:
                    st.error(f"The recommendation service returned an unexpected error ({response.status_code}). Please try again later.")

            except httpx.TimeoutException:
                # Friendly message when the service takes too long (AC5)
                st.error(
                    "⏳ The recommendation service is taking too long to respond. "
                    "Please check your connection or try again in a few moments."
                )

            except httpx.ConnectError:
                # Friendly message when the service is down / not available (AC5)
                st.error(
                    f"🔌 The recommendation service is currently not available at {service_url}. "
                    "Please make sure the backend service is running (`uvicorn api.main:app`) and try again."
                )

            except Exception as exc:
                st.error(f"⚠️ Unable to reach the recommendation service right now: {exc}")


if __name__ == "__main__":
    main()
