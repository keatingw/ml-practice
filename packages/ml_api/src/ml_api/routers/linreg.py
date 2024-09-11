"""Router for Linear Regression implementations."""

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from py_algos.linreg.native_py import LinearRegressionGD
from py_algos.linreg.numpy_py import LinearRegressionNumpy
from rs_algos import LinRegGDRust


class LinRegInput(BaseModel):
    """Input for linear regression model routes."""

    intercept: bool = True
    data: list[list[float]]
    target: list[float]


class LinRegOutput(BaseModel):
    """Output for linear regression model routes."""

    coefficients: list[float]
    predictions: list[float]


router = APIRouter(
    prefix="/linreg",
    tags=["linreg"],
)


@router.post("/native_py")
def linreg_native_py(
    linreg_input: LinRegInput,
) -> LinRegOutput:
    """Linear regression endpoint for native Python."""
    lr_py = LinearRegressionGD(intercept=linreg_input.intercept)
    pred_py = lr_py.fit_predict(linreg_input.data, linreg_input.target)
    return LinRegOutput(coefficients=lr_py.weights, predictions=pred_py)


@router.post("/numpy_py")
def linreg_numpy_py(
    linreg_input: LinRegInput,
) -> LinRegOutput:
    """Linear regression endpoint for numpy."""
    lr_numpy = LinearRegressionNumpy(intercept=linreg_input.intercept)
    pred_numpy = lr_numpy.fit_predict(
        np.array(linreg_input.data), np.array(linreg_input.target)
    )
    return LinRegOutput(
        coefficients=lr_numpy.weights.tolist(), predictions=pred_numpy.tolist()
    )


@router.post("/rs")
def linreg_rs(
    linreg_input: LinRegInput,
) -> LinRegOutput:
    """Linear regression endpoint for Rust."""
    lr_rs = LinRegGDRust(intercept=linreg_input.intercept)
    lr_rs.fit(linreg_input.data, linreg_input.target)
    assert lr_rs.weights is not None  # noqa: S101
    pred_rs = lr_rs.predict(linreg_input.data)
    return LinRegOutput(coefficients=lr_rs.weights, predictions=pred_rs)
