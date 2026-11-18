"""Request and response formats of the recommendation service (US6.1 / US6.2)."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """What the client sends to POST /recommend. Every field is optional."""

    user_id: int | None = Field(
        default=None,
        ge=1,
        description="MovieLens user id. Leave it out for a new user with no history.",
    )
    genre: str | None = Field(
        default=None,
        description="Only recommend movies of this genre, e.g. 'Comedy'.",
    )
    mood: str | None = Field(
        default=None,
        description="Only recommend movies that fit this mood, e.g. 'relaxed' (see GET /moods). Don't send it together with genre.",
    )
    n: int = Field(default=5, ge=1, le=50, description="How many movies to return.")


class MovieOut(BaseModel):
    """One recommended movie."""

    rank: int = Field(description="Position in the list, starting at 1.")
    movie_id: int
    title: str
    genres: list[str]
    score: float = Field(description="Popularity score: higher means more recommended.")


class RecommendResponse(BaseModel):
    """What the service answers: the movies ordered from most to least recommended."""

    user_id: int | None
    genre: str | None
    mood: str | None = None
    source: str = Field(description="Which model produced the list.")
    recommendations: list[MovieOut]


# Examples shown in the documentation page (/docs)
REQUEST_EXAMPLES = {
    "known_user": {
        "summary": "Known user",
        "value": {"user_id": 1, "n": 5},
    },
    "genre_new_user": {
        "summary": "New user who wants a genre",
        "value": {"genre": "Comedy", "n": 5},
    },
    "mood_new_user": {
        "summary": "New user in a mood",
        "value": {"mood": "relaxed", "n": 5},
    },
    "known_user_and_genre": {
        "summary": "Known user + genre",
        "value": {"user_id": 1, "genre": "Drama", "n": 5},
    },
}

EXAMPLE_RESPONSE = {
    "user_id": 1,
    "genre": None,
    "source": "popularity-baseline",
    "recommendations": [
        {
            "rank": 1,
            "movie_id": 318,
            "title": "Shawshank Redemption, The (1994)",
            "genres": ["Drama"],
            "score": 4.52,
        },
        {
            "rank": 2,
            "movie_id": 858,
            "title": "Godfather, The (1972)",
            "genres": ["Action", "Crime", "Drama"],
            "score": 4.50,
        },
        {
            "rank": 3,
            "movie_id": 50,
            "title": "Usual Suspects, The (1995)",
            "genres": ["Crime", "Thriller"],
            "score": 4.49,
        },
        {
            "rank": 4,
            "movie_id": 527,
            "title": "Schindler's List (1993)",
            "genres": ["Drama", "War"],
            "score": 4.48,
        },
        {
            "rank": 5,
            "movie_id": 1198,
            "title": "Raiders of the Lost Ark (1981)",
            "genres": ["Action", "Adventure"],
            "score": 4.46,
        },
    ],
}
