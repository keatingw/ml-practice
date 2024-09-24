"""FastAPI app definition module."""

from fastapi import FastAPI

from ml_api.routers.kmeans import router as kmeans_router
from ml_api.routers.linreg import router as linreg_router

app = FastAPI()
app.include_router(kmeans_router)
app.include_router(linreg_router)
