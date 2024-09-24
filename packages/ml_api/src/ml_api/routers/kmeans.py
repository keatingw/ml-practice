"""Router for KMeans implementations."""

from random import Random

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from py_algos.kmeans.native_py import KMeans
from py_algos.kmeans.numpy_py import KMeansNumpy
from rs_algos import KMeansRust


class KMeansInput(BaseModel):
    """Input for KMeans model routes."""

    seed: int
    n_clusters: int
    n_iter: int
    data: list[list[float]]


class KMeansOutput(BaseModel):
    """Output for KMeans model routes."""

    clusters: list[int]


router = APIRouter(
    prefix="/kmeans",
    tags=["kmeans"],
)


@router.post("/native_py")
def kmeans_native_py(kmeans_input: KMeansInput) -> KMeansOutput:
    """K-means endpoint for native Python."""
    km_py = KMeans(Random(kmeans_input.seed), kmeans_input.n_clusters)  # noqa: S311
    km_py.fit(kmeans_input.data)
    return KMeansOutput(clusters=[i[1] for i in km_py._datapoints])


@router.post("/numpy_py")
def kmeans_numpy_py(kmeans_input: KMeansInput) -> KMeansOutput:
    """K-means endpoint for numpy."""
    km_numpy = KMeansNumpy(kmeans_input.n_clusters, kmeans_input.n_iter)
    pred_numpy = km_numpy.fit_predict(np.array(kmeans_input.data))
    return KMeansOutput(clusters=pred_numpy.tolist())


@router.post("/rs")
def kmeans_rs(kmeans_input: KMeansInput) -> KMeansOutput:
    """K-means regression endpoint for Rust."""
    km_rs = KMeansRust(kmeans_input.n_clusters, kmeans_input.n_iter, kmeans_input.seed)
    km_rs.fit(kmeans_input.data)
    assert km_rs.allocations is not None  # noqa: S101
    return KMeansOutput(clusters=km_rs.allocations)
