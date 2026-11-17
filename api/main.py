"""Recommendation service (US6.1). Start it with:  uvicorn api.main:app --reload"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Request

from api.schemas import (
    EXAMPLE_RESPONSE,
    REQUEST_EXAMPLES,
    RecommendRequest,
    RecommendResponse,
)
from api.service import (
    RecommenderService,
    UnknownGenreError,
    UnknownMoodError,
    UserNotFoundError,
)


def create_app(service: RecommenderService | None = None) -> FastAPI:
    """Build the app. Tests pass their own small service; normally the data is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = (
            service if service is not None else RecommenderService.from_env()
        )
        yield

    app = FastAPI(
        title="Movie recommender API",
        version="0.3.0",
        description="Ask for movie recommendations for a user, a genre, or both.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        """Quick check that the service is running, and which model it has loaded."""
        return {"status": "ok", "model": request.app.state.service.model_info}

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        responses={
            200: {"content": {"application/json": {"example": EXAMPLE_RESPONSE}}},
            404: {"description": "The user is not in the system"},
            422: {
                "description": "The request is not valid: a genre or mood that doesn't exist, or both sent together"
            },
        },
    )
    def recommend(
        request: Request,
        req: Annotated[RecommendRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
    ) -> RecommendResponse:
        """Return movies ordered from most to least recommended.

        - **user_id**: a known user gets movies they haven't rated yet. Leave it out for a new user.
        - **genre**: only movies of that genre (e.g. Comedy, Drama, Action).
        - **mood**: only movies that fit a mood (e.g. relaxed). Send a genre or a mood, not both.
        """
        svc: RecommenderService = request.app.state.service
        if req.genre and req.mood:
            raise HTTPException(
                status_code=422,
                detail="Send either a genre or a mood, not both.",
            )
        try:
            genre = svc.canonical_genre(req.genre) if req.genre else None
            mood = svc.canonical_mood(req.mood) if req.mood else None
            movies = svc.recommend(user_id=req.user_id, genre=genre, mood=mood, n=req.n)
        except UserNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"User {req.user_id} is not in the system."
            )
        except UnknownMoodError:
            raise HTTPException(
                status_code=422,
                detail=f"Mood '{req.mood}' doesn't exist. Valid moods: {', '.join(svc.moods)}.",
            )
        except UnknownGenreError:
            raise HTTPException(
                status_code=422,
                detail=f"Genre '{req.genre}' doesn't exist. Valid genres: {', '.join(svc.genres)}.",
            )
        return RecommendResponse(
            user_id=req.user_id,
            genre=genre,
            mood=mood,
            source=svc.source,
            recommendations=movies,
        )

    @app.get("/genres")
    def genres(request: Request) -> dict:
        """The list of genres you can ask for."""
        return {"genres": request.app.state.service.genres}

    @app.get("/moods")
    def moods(request: Request) -> dict:
        """The moods you can ask for, and the genres linked to each one (from params.yaml)."""
        return {"moods": request.app.state.service.moods}

    return app


app = create_app()
