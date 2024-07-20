"""Implementations for Linear Regression."""

from .native_py import LinearRegressionGD
from .numpy_py import LinearRegressionNumpy

__all__ = [
    "LinearRegressionGD",
    "LinearRegressionNumpy",
]
