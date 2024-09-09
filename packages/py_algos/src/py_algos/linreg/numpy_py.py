"""Practice Linear Regression implementation with Numpy."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class LinearRegressionNumpy:
    """Numpy-based Linear Regression using matrix inversion for analytical solution."""

    intercept: bool = True
    data: npt.NDArray[np.floating[Any]] = field(init=False, repr=False)
    weights: npt.NDArray[np.floating[Any]] = field(init=False, repr=False)

    def fit(
        self,
        data: npt.NDArray[np.floating[Any]],
        target: npt.NDArray[np.floating[Any]],
    ) -> None:
        """Fits the weights with analytical solution to OLS."""
        self.data = data.copy()
        self.target = target.copy()
        if self.intercept:
            self.data = np.concatenate(
                [np.ones((self.data.shape[0], 1)), self.data], axis=1
            )
        self.weights = (
            np.linalg.inv(self.data.transpose() @ self.data)
            @ self.data.transpose()
            @ self.target
        )

    def fit_predict(
        self,
        data: npt.NDArray[np.floating[Any]],
        target: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Fits the weights with analytical solution to OLS and returns prediction."""
        self.fit(data, target)
        return self.data @ self.weights
