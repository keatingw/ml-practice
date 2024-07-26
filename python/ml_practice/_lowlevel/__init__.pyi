from typing import Self

def hello() -> str: ...

class KMeansRust:
    centers: list[list[float]]
    allocations: list[int]
    max_iter: int

    def __new__(
        cls, num_centers: int, max_iter: int, data: list[list[float]], seed: int
    ) -> Self: ...
    def fit(self, data: list[list[float]]) -> None: ...

class LinRegGDRust:
    lr: float
    intercept: bool
    num_iter: int
    weights: list[float] | None

    def __new__(
        cls,
        lr: float = 0.01,
        intercept: bool = True,  # noqa: FBT001, FBT002
        num_iter: int = 100,
    ) -> Self: ...
    def fit(self, data: list[list[float]], target: list[float]) -> None: ...
