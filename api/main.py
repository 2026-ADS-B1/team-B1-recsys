"""Recommendation service (US6.1). Start it with:  uvicorn api.main:app --reload"""

from typing import Annotated

from fastapi import Body, FastAPI

from api.schemas import (
    EXAMPLE_RESPONSE,
    REQUEST_EXAMPLES,
    RecommendRequest,
    RecommendResponse,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie recommender API",
        version="0.1.0",
        description="Ask for movie recommendations for a user, a genre, or both.",
    )

    @app.get("/health")
    def health() -> dict:
        """Quick check that the service is running."""
        return {"status": "ok"}

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        responses={
            200: {"content": {"application/json": {"example": EXAMPLE_RESPONSE}}}
        },
    )
    def recommend(
        req: Annotated[RecommendRequest, Body(openapi_examples=REQUEST_EXAMPLES)],
    ) -> RecommendResponse:
        """Return movies ordered from most to least recommended.

        For now this is a fixed example answer (task #25); real results come in task #27.
        """
        return RecommendResponse(**EXAMPLE_RESPONSE)

    return app


app = create_app()
