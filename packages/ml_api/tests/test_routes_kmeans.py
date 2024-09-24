"""Tests for K-means routes."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_api.routers.kmeans import KMeansInput


@pytest.fixture(scope="module")
def kmeans_input() -> KMeansInput:
    """K-means API input fixture."""
    data = np.random.normal(0, 2, (1_000, 15))  # noqa: NPY002
    return KMeansInput(seed=0, n_clusters=5, n_iter=100, data=data.tolist())


def test_kmeans_native(test_client: TestClient, kmeans_input: KMeansInput) -> None:
    """Test for native Python K-means endpoint."""
    response = test_client.post("/kmeans/native_py", json=kmeans_input.model_dump())
    assert response.is_success
    assert len(response.json()["clusters"]) == len(kmeans_input.data)


def test_kmeans_numpy(test_client: TestClient, kmeans_input: KMeansInput) -> None:
    """Test for numpy K-means endpoint."""
    response = test_client.post("/kmeans/numpy_py", json=kmeans_input.model_dump())
    assert response.is_success
    assert len(response.json()["clusters"]) == len(kmeans_input.data)


def test_kmeans_rust(test_client: TestClient, kmeans_input: KMeansInput) -> None:
    """Test for Rust K-means endpoint."""
    response = test_client.post("/kmeans/rs", json=kmeans_input.model_dump())
    assert response.is_success
    assert len(response.json()["clusters"]) == len(kmeans_input.data)
