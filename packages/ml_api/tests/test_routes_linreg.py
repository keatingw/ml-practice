"""Tests for linear regression routes."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_api.routers.linreg import LinRegGDInput, LinRegInput

COEFFICIENTS = (11, 1, 3, -4, 2.1)
RTOL = 0.05


@pytest.fixture(scope="module")
def linreg_data() -> tuple[list[list[float]], list[float]]:
    """Fixture for linear regression input data."""
    data = np.random.normal(0, 2, (1_000, 4))  # noqa: NPY002
    target = (
        COEFFICIENTS[0]
        + COEFFICIENTS[1] * data[:, 0]
        + COEFFICIENTS[2] * data[:, 1]
        + COEFFICIENTS[3] * data[:, 2]
        + COEFFICIENTS[4] * data[:, 3]
        + np.random.normal(0, 1, 1_000)  # noqa: NPY002
    )
    return data.tolist(), target.tolist()


@pytest.fixture(scope="module")
def linreg_input(linreg_data: tuple[list[list[float]], list[float]]) -> LinRegInput:
    """Linear regression API input fixture."""
    return LinRegInput(intercept=True, data=linreg_data[0], target=linreg_data[1])


@pytest.fixture(scope="module")
def linreg_gd_input(
    linreg_data: tuple[list[list[float]], list[float]],
) -> LinRegGDInput:
    """Linear regression by GD API input fixture."""
    return LinRegGDInput(
        intercept=True,
        data=linreg_data[0],
        target=linreg_data[1],
        lr=0.01,
        num_iter=1_000,
    )


def test_linreg_native(test_client: TestClient, linreg_gd_input: LinRegGDInput) -> None:
    """Test for native Python linear regression endpoint."""
    response = test_client.post("/linreg/native_py", json=linreg_gd_input.model_dump())
    assert response.is_success
    resp_json = response.json()
    assert len(resp_json["predictions"]) == len(linreg_gd_input.data)
    assert np.allclose(COEFFICIENTS, resp_json["coefficients"], rtol=RTOL)


def test_kmeans_numpy(test_client: TestClient, linreg_input: LinRegInput) -> None:
    """Test for numpy linear regression endpoint."""
    response = test_client.post("/linreg/numpy_py", json=linreg_input.model_dump())
    assert response.is_success
    resp_json = response.json()
    assert len(resp_json["predictions"]) == len(linreg_input.data)
    assert np.allclose(COEFFICIENTS, resp_json["coefficients"], rtol=RTOL)


def test_linreg_rust(test_client: TestClient, linreg_gd_input: LinRegGDInput) -> None:
    """Test for Rust linear regression endpoint."""
    response = test_client.post("/linreg/rs", json=linreg_gd_input.model_dump())
    assert response.is_success
    resp_json = response.json()
    assert len(resp_json["predictions"]) == len(linreg_gd_input.data)
    assert np.allclose(COEFFICIENTS, resp_json["coefficients"], rtol=RTOL)
