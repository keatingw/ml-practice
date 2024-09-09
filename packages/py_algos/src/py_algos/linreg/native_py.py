"""Practice Linear Regression implementation in native Python."""

from dataclasses import dataclass, field

from py_algos.utils import matvecmul


@dataclass
class LinearRegressionGD:
    """Implements Linear regression in native python using gradient descent."""

    intercept: bool = True
    max_iter: int = 100
    lr: float = 0.01
    weights: list[float] = field(init=False, repr=False)

    def predict(self, data: list[list[float]]) -> list[float]:
        """Predicts outputs on data, adding constants if intercept is True."""
        if self.intercept:
            data = [[1.0, *i] for i in data]

        return matvecmul(data, self.weights)

    def fit(self, data: list[list[float]], target: list[float]) -> None:
        """Fits using gradient descent."""
        if self.intercept:
            data = [[1.0, *i] for i in data]

        m = len(data)
        self.weights = [1.0 for _ in range(len(data[0]))]
        for _ in range(self.max_iter):
            pred = matvecmul(data, self.weights)
            grad = matvecmul(
                list(zip(*data)),  # type: ignore[arg-type]
                [i - j for i, j in zip(pred, target)],
            )
            for w in range(len(self.weights)):
                self.weights[w] -= grad[w] * self.lr / m

    def fit_predict(self, data: list[list[float]], target: list[float]) -> list[float]:
        """Fits and predicts on data."""
        self.fit(data, target)
        return self.predict(data)
