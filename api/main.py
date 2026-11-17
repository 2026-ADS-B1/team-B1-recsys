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
from api.service import RecommenderService, UnknownGenreError, UserNotFoundError


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
    def health() -> dict:
        """Quick check that the service is running."""
        return {"status": "ok"}

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        responses={
            200: {"content": {"application/json": {"example": EXAMPLE_RESPONSE}}},
            404: {"description": "The user is not in the system"},
            422: {
                "description": "The request is not valid (for example, a genre that doesn't exist)"
            },
            501: {"description": "Mood is not available yet"},
        },
    )
    def recommend(
        request: Request,
        req: Annotated[RecommendRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
    ) -> RecommendResponse:
        """Return movies ordered from most to least recommended.

        - **user_id**: a known user gets movies they haven't rated yet. Leave it out for a new user.
        - **genre**: only movies of that genre (e.g. Comedy, Drama, Action).
        """
        svc: RecommenderService = request.app.state.service
        if req.mood is not None:
            raise HTTPException(
                status_code=501, detail="Mood is not available yet. Use genre instead."
            )
        try:
            genre = svc.canonical_genre(req.genre) if req.genre else None
            movies = svc.recommend(user_id=req.user_id, genre=genre, n=req.n)
        except UserNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"User {req.user_id} is not in the system."
            )
        except UnknownGenreError:
            raise HTTPException(
                status_code=422,
                detail=f"Genre '{req.genre}' doesn't exist. Valid genres: {', '.join(svc.genres)}.",
            )
        return RecommendResponse(
            user_id=req.user_id,
            genre=genre,
            source=svc.source,
            recommendations=movies,
        )

    @app.get("/genres")
    def genres(request: Request) -> dict:
        """The list of genres you can ask for."""
        return {"genres": request.app.state.service.genres}

    return app


app = create_app()
