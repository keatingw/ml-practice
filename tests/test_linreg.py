"""Tests for linear regression."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from py_algos.linreg import LinearRegressionGD, LinearRegressionNumpy
from rs_algos import LinRegGDRust
from sklearn.linear_model import LinearRegression


@pytest.fixture
def linreg_input() -> (
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
):
    """Inputs for linear regressino testing."""
    rng = np.random.default_rng(1)
    return rng.normal(0, 10, (1_000, 4)), rng.normal(0, 1, (1_000))


def test_linreg_outputs(
    linreg_input: tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]],
) -> None:
    """Test linear regression manual outputs vs sklearn implementation."""
    lr_sk = LinearRegression()
    lr_np = LinearRegressionNumpy()
    lr_py = LinearRegressionGD(max_iter=10_000, lr=0.01)
    lr_rs = LinRegGDRust(lr=0.01, num_iter=10_000)

    lr_sk.fit(linreg_input[0], linreg_input[1])
    lr_np.fit(linreg_input[0], linreg_input[1])
    lr_py.fit(linreg_input[0].tolist(), linreg_input[1].tolist())
    lr_rs.fit(linreg_input[0].tolist(), linreg_input[1].tolist())

    skarr = np.array([lr_sk.intercept_, *lr_sk.coef_])
    nparr = lr_np.weights
    pyarr = np.array(lr_py.weights)
    rsarr = np.array(lr_rs.weights)

    assert np.allclose(
        skarr, nparr
    ), "sklearn weights different to python numpy weights"
    assert np.allclose(
        skarr, pyarr
    ), "sklearn weights different to native python weights"
    assert np.allclose(skarr, rsarr), "sklearn weights different to rust weights"
