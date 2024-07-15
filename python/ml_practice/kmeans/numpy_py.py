"""Practice KMeans implementation with Numpy."""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris  # type: ignore[import-untyped]

data = load_iris()["data"]


class KMeansNumpy:
    """Numpy-based KMeans algorithm implementation."""

    centers: npt.NDArray[np.floating[Any]]
    datapoints: npt.NDArray[np.floating[Any]]
    n_clusters: int
    n_iter: int | None
    assigned_clusters: npt.NDArray[np.floating[Any]]

    def __init__(
        self,
        n_clusters: int = 5,
        n_iter: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_iter = n_iter

    def fit(self, data: npt.NDArray[np.floating[Any]]) -> None:
        """Fits the data by computing the clusters."""
        self.datapoints = data
        self.n_vars = data.shape[1]

        # start centers at random set of points
        self.centers = data[
            np.random.choice(  # noqa: NPY002
                np.arange(data.shape[0]),
                size=self.n_clusters,
                replace=False,
            ),
            :,
        ]
        self.assigned_clusters = np.zeros(data.shape[0])
        itercount = 0
        while self.n_iter is None or itercount < self.n_iter:
            itercount += 1
            print(f"Iteration {itercount}")  # noqa: T201
            prev_assigned = self.assigned_clusters.copy()
            self.assigned_clusters = (
                ((data - self.centers.reshape(-1, 1, self.n_vars)) ** 2)
                .sum(-1)
                .argmin(axis=0)
            )
            # once converged, stop
            if np.array_equal(prev_assigned, self.assigned_clusters):
                break
            for i in np.arange(self.n_clusters):
                self.centers[i, :] = self.datapoints[self.assigned_clusters == i].mean(
                    axis=0
                )

    def fit_predict(
        self, data: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Fits clusters to data and returns cluster predictions."""
        self.fit(data)
        return self.assigned_clusters
