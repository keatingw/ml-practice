"""Practice implementation of K-means in native Python."""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast


@dataclass
class KMeans:
    """Class to implement K-means clustering."""

    rng: random.Random
    n_clusters: int
    _datapoints: list[tuple[list[float], int]] = field(default_factory=list, init=False)
    """Maps datapoints to clusters."""

    def fit(self, points: list[list[float]]) -> None:
        """Fits k-means clusters for a set of input points."""
        initial_assignments = self.rng.choices(range(self.n_clusters), k=len(points))
        self._datapoints = list(zip(points, initial_assignments))
        while True:
            # get original clusters
            pre_assignments = [i[1] for i in self._datapoints]
            # do a fit loop
            self._fit_loop()
            # get new assignments
            new_assignments = [i[1] for i in self._datapoints]
            if new_assignments == pre_assignments:
                break

    def _fit_loop(self) -> None:
        """Interior loop operation to reassign points."""
        # Copy cluster centroids so they don't shift per loop
        original_centres = deepcopy(self.cluster_centres)
        for idx, i in enumerate(self._datapoints):
            # find best cluster and replace with it per datapoint
            distances = [
                (cluster, euclidean_distance(i[0], centroid))
                for cluster, centroid in original_centres.items()
            ]
            min_clust = min(distances, key=lambda x: x[1])
            self._datapoints[idx] = (i[0], min_clust[0])

    @property
    def cluster_centres(self) -> dict[int, list[float]]:
        """Gets centroids for clusters as a dict."""
        return {
            i: get_centroid([p[0] for p in self._datapoints if p[1] == i])
            for i in range(self.n_clusters)
        }


def get_centroid(cluster: list[list[float]]) -> list[float]:
    """Calculates the mean of a set of points as its centroid."""
    return [sum(i) / len(i) for i in zip(*cluster)]


def euclidean_distance(p1: list[float], p2: list[float]) -> float:
    """Calculates euclidean distance between two points."""
    return cast(
        float, sum((i1 - i2) ** 2 for i1, i2 in zip(p1, p2, strict=True)) ** 0.5
    )
