"""Implementations for Linear Regression."""

from ml_practice._lowlevel import LinRegGDRust

from .native_py import LinearRegressionGD
from .numpy_py import LinearRegressionNumpy

__all__ = [
    "LinearRegressionGD",
    "LinearRegressionNumpy",
    "LinRegGDRust",
]
