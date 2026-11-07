"""Chapter 01: quadratic forms and linear least squares (lecture C1).

Notation (see ``book/notation.md``):

- quadratic form ``q(x) = x^T W x + L x + c``; ``W`` is called ``Q`` in the lectures;
- linear model ``y = M p + noise``, criterion ``j(p) = ||M p - y||^2``,
  estimate ``p_hat = (M^T M)^{-1} M^T y``.

Vectors are 1-D arrays of shape ``(n,)``. ``L`` is the row vector of the linear term, also
stored as a 1-D array of shape ``(n,)``.
"""

import numpy as np


def quadratic_form(x: np.ndarray, W: np.ndarray, L: np.ndarray, c: float) -> float:
    """Evaluate ``q(x) = x^T W x + L x + c``.

    Parameters
    ----------
    x : (n,) array
    W : (n, n) array, not necessarily symmetric
    L : (n,) array
    c : float

    Returns
    -------
    float
    """
    q = x.transpose() @ W @ x + L @ x + c
    return q


def quadratic_gradient(x: np.ndarray, W: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Gradient of ``q`` at ``x``, as an ``(n,)`` array.

    Valid for any square ``W``, symmetric or not. Hint: write ``q`` with the symmetric part
    ``(W + W^T) / 2`` first.
    """
    gradient = (W + W.transpose()) @ x + L.transpose()
    return gradient


def quadratic_minimizer(W: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Minimizer ``x*`` of ``q``, as an ``(n,)`` array.

    ``W`` must be symmetric positive definite; otherwise ``q`` has no unique minimum and the
    function raises ``ValueError``. Solve a linear system; do not form ``W^{-1}``.
    """
    if not np.all(np.linalg.eigvals(W) > 0):
        raise ValueError("W must be symmetric positive definite.")
    x_star = np.linalg.solve(2 * W, -L)
    return x_star


def polynomial_regressor(t: np.ndarray, degree: int) -> np.ndarray:
    """Regressor matrix ``M`` of a polynomial model, shape ``(N, degree + 1)``.

    Columns are in decreasing powers, ``[t^d, ..., t, 1]``, so that ``M @ p`` evaluates the
    polynomial ``p[0] t^d + ... + p[d]`` at every ``t`` (same order as ``numpy.polyval``).
    """
    N = t.shape[0]
    M = np.zeros((N, degree + 1))
    for i in range(degree + 1):
        M[:, i] = t ** (degree - i)
    return M


def least_squares(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares estimate ``p_hat = (M^T M)^{-1} M^T y``, shape ``(n,)``.

    ``M`` has shape ``(N, n)`` with full column rank; ``y`` has shape ``(N,)``.
    Implement it from the normal equations (``numpy.linalg.lstsq`` is only for checking).
    """
    MtM = M.transpose() @ M
    Mty = M.transpose() @ y
    p_hat = np.linalg.solve(MtM, Mty)
    return p_hat


def residuals(M: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Residual vector ``r = y - M p``, shape ``(N,)``."""
    r = y - M @ p
    return r


def criterion(M: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    """Least-squares criterion ``j(p) = ||M p - y||^2``."""
    r = residuals(M, y, p)
    j = r.transpose() @ r
    return j
