from typing import Final, Self

def hello() -> str: ...

class KMeansRust:
    num_centers: Final[int]
    max_iter: Final[int]
    seed: Final[int]
    centers: Final[list[list[float]]]
    allocations: Final[list[int] | None]

    # pyo3 doesn't actually support init, but this allows documentation
    def __init__(self, num_centers: int, max_iter: int, seed: int) -> None: ...
    def __new__(cls, num_centers: int, max_iter: int, seed: int) -> Self: ...
    def fit(self, data: list[list[float]]) -> None: ...

class LinRegGDRust:
    lr: Final[float]
    intercept: Final[bool]
    num_iter: Final[int]
    weights: Final[list[float] | None]

    def __init__(
        self,
        lr: float = 0.01,
        intercept: bool = True,  # noqa: FBT001, FBT002
        num_iter: int = 100,
    ) -> None: ...
    def __new__(
        cls,
        lr: float = 0.01,
        intercept: bool = True,  # noqa: FBT001, FBT002
        num_iter: int = 100,
    ) -> Self: ...
    def fit(self, data: list[list[float]], target: list[float]) -> None: ...
    def predict(self, data: list[list[float]]) -> list[float]: ...
