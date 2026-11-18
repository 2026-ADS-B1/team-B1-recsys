"""Recommendations web page (US 7.2).

Start with:
    streamlit run app/main.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from api.config import load_settings
from api.data import load_movies
from api.schemas import EXAMPLE_RESPONSE

DEFAULT_MOVIES_PATH = Path("data/raw/ml-1m/movies.dat")


@st.cache_data
def get_moods() -> list[str]:
    """Read available moods dynamically from params.yaml (US6.2 configuration)."""
    settings = load_settings()
    return sorted(list(settings.moods.keys()))


@st.cache_data
def get_genres() -> list[str]:
    """Read available genres from the movie catalogue."""
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

    # Standard MovieLens 1M catalogue genres fallback if data not pulled yet
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
        genres = get_genres()
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
        st.subheader("Recommended for You Tonight")
        
        # Summary of the request
        user_display = "Guest" if is_guest else f"User {user_id}"
        filter_display = (
            f"Genre: **{selected_genre}**"
            if selected_genre
            else (f"Mood: **{selected_mood.capitalize()}**" if selected_mood else "None")
        )
        st.caption(f"Profile: **{user_display}** | Filter: {filter_display}")

        # Example recommendations (mocked for now, as specified in the task)
        recommendations = EXAMPLE_RESPONSE.get("recommendations", [])
        
        table_data = []
        for rec in recommendations:
            genres_formatted = ", ".join(rec.get("genres", []))
            table_data.append(
                {
                    "Rank": rec.get("rank"),
                    "Title": rec.get("title"),
                    "Genres": genres_formatted,
                    "Score": f"{rec.get('score'):.2f}",
                }
            )

        df_display = pd.DataFrame(table_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
