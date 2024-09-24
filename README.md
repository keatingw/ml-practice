# ml-practice

A workspace structured project for revising machine learning algorithms.

The top-level workspace defines the development dependencies and any cross-cutting
tests. Underlying this there are 3 packages:

1. `py_algos`: this is a pure Python package for machine learning algorithm
implementations. Generally, this will include both a native version (using only
standard library objects), and a `numpy`-based version with matrix operations.
2. `rs_algos`: this is a Rust extension package, with parallel machine learning
algorithm implementations exposed to Python through maturin/PyO3.
3. `ml_api`: this is a FastAPI server, to provide a convenient interface to test
each of the 3 parallel implementations for a given algorithm.
