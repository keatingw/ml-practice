"""Configuration for API testing."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """Fixture for FastAPI app test client."""
    from ml_api.app import app

    return TestClient(app)
