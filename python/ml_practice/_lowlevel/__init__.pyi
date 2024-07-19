from typing import Self

def hello() -> str: ...

class KMeansRust:
    centers: list[list[float]]
    allocations: list[int]
    max_iter: int

    def __new__(
        cls, num_centers: int, max_iter: int, data: list[list[float]], seed: int
    ) -> Self: ...
