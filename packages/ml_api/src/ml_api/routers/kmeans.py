"""Router for KMeans implementations."""

from fastapi import APIRouter
from pydantic import BaseModel


class KMeansInput(BaseModel):
    """Input for KMeans model routes."""


class KMeansOutput(BaseModel):
    """Output for KMeans model routes."""


router = APIRouter(
    prefix="/kmeans",
    tags=["kmeans"],
)
