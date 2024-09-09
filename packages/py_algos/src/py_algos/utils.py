"""Manual utilities for implementing ML algorithms."""


def matmul(
    l: list[list[float]],  # noqa: E741
    r: list[list[float]],
) -> list[list[float]]:
    """List-based matrix multiplications."""
    l_i, l_j = len(l), len(l[0])
    r_i, r_j = len(r), len(r[0])

    # check for compatible sizes
    if l_j != r_i:
        msg = f"Shape mismatch ({l_i}, {l_j}), ({r_i}, {r_j})"
        raise ValueError(msg)

    out = [[0.0 for _ in range(r_j)] for _ in range(l_i)]

    for rowidx in range(l_i):
        rowvec = l[rowidx]
        for colidx in range(r_j):
            out[rowidx][colidx] = sum(
                x * y for x, y in zip(rowvec, [i[colidx] for i in r], strict=True)
            )

    return out


def matvecmul(
    l: list[list[float]],  # noqa: E741
    r: list[float],
) -> list[float]:
    """Matrix-vector multiplication, treating the vector as a RHS column vec and flattening the output."""
    matrix_out = matmul(l, [[i] for i in r])
    return [i[0] for i in matrix_out]
