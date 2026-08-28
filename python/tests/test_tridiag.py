"""Thomas solver vs dense/banded references, plus input validation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_banded  # reference only — never in the pricer path

from localvol import thomas_solve


def _dense(sub, diag, sup):
    n = diag.size
    a = np.diag(diag)
    if n > 1:
        a += np.diag(sub, -1) + np.diag(sup, 1)
    return a


def test_matches_dense_solve_on_random_systems():
    """Property-style: random diagonally-dominant systems of many sizes."""
    rng = np.random.default_rng(123)
    for n in [1, 2, 3, 5, 17, 64, 201]:
        for _ in range(3):
            sub = rng.uniform(-1.0, 1.0, max(n - 1, 0))
            sup = rng.uniform(-1.0, 1.0, max(n - 1, 0))
            diag = rng.uniform(2.5, 4.0, n) * rng.choice([-1.0, 1.0], n)
            rhs = rng.uniform(-5.0, 5.0, n)
            x = thomas_solve(sub, diag, sup, rhs)
            x_ref = np.linalg.solve(_dense(sub, diag, sup), rhs)
            np.testing.assert_allclose(x, x_ref, rtol=1e-12, atol=1e-12)


def test_matches_scipy_banded():
    rng = np.random.default_rng(7)
    n = 120
    sub = rng.normal(size=n - 1)
    sup = rng.normal(size=n - 1)
    diag = 4.0 + rng.uniform(0.0, 1.0, n)
    rhs = rng.normal(size=n)
    ab = np.zeros((3, n))
    ab[0, 1:] = sup
    ab[1, :] = diag
    ab[2, :-1] = sub
    x_ref = solve_banded((1, 1), ab, rhs)
    np.testing.assert_allclose(thomas_solve(sub, diag, sup, rhs), x_ref, rtol=1e-11)


def test_identity_and_size_one():
    np.testing.assert_allclose(
        thomas_solve(np.zeros(2), np.ones(3), np.zeros(2), np.array([1.0, 2.0, 3.0])),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(thomas_solve(np.zeros(0), np.array([2.0]), np.zeros(0), np.array([8.0])), [4.0])


@pytest.mark.parametrize(
    "sub,diag,sup,rhs",
    [
        (np.zeros(1), np.ones(3), np.zeros(2), np.ones(3)),  # bad sub length
        (np.zeros(2), np.ones(3), np.zeros(2), np.ones(4)),  # bad rhs length
        (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)),  # empty
    ],
)
def test_shape_validation(sub, diag, sup, rhs):
    with pytest.raises(ValueError):
        thomas_solve(sub, diag, sup, rhs)


def test_rejects_nonfinite_and_zero_pivot():
    with pytest.raises(ValueError, match="non-finite"):
        thomas_solve(np.zeros(1), np.array([1.0, np.nan]), np.zeros(1), np.ones(2))
    with pytest.raises(ValueError, match="pivot"):
        thomas_solve(np.zeros(1), np.array([0.0, 1.0]), np.zeros(1), np.ones(2))
    # elimination-induced zero pivot: [[1,1],[1,1]] is singular
    with pytest.raises(ValueError, match="pivot"):
        thomas_solve(np.array([1.0]), np.array([1.0, 1.0]), np.array([1.0]), np.ones(2))
