"""Chapter 01: noisy measurements of a polynomial."""

import numpy as np

from kalman.least_squares import polynomial_regressor


def simulate(p: np.ndarray, t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Measure the polynomial with coefficients ``p`` at times ``t``.

    Returns ``y`` of shape ``(N,)`` with ``y_i = p[0] t_i^d + ... + p[d] + beta_i`` and
    ``beta_i ~ N(0, sigma^2)`` independent. Coefficients are in decreasing powers, as in
    ``kalman.least_squares.polynomial_regressor``. ``sigma = 0`` gives exact values.
    """
    M = polynomial_regressor(t, len(p) - 1)
    y_true = M @ p
    beta = rng.normal(0, sigma, size=t.shape)
    y = y_true + beta
    return y
