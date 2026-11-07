"""Chapter 01: quadratic forms and linear least squares (lecture C1)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kalman import plot
from kalman.least_squares import (
    criterion,
    least_squares,
    polynomial_regressor,
    quadratic_form,
    quadratic_gradient,
    quadratic_minimizer,
    residuals,
)
from kalman.models.polynomial import simulate

# A small line fit computed by hand:
# t = (0, 1, 2), y = (0, 1, 1), model y = p1 t + p2
# M^T M = [[5, 3], [3, 3]], M^T y = (3, 2)  ->  p_hat = (1/2, 1/6)
# y_hat = (1/6, 2/3, 7/6), r = (-1/6, 1/3, -1/6), j(p_hat) = 1/6
T_HAND = np.array([0.0, 1.0, 2.0])
Y_HAND = np.array([0.0, 1.0, 1.0])
M_HAND = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
P_HAND = np.array([1 / 2, 1 / 6])
R_HAND = np.array([-1 / 6, 1 / 3, -1 / 6])


def random_spd(n, rng):
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def numerical_gradient(fun, x, h=1e-6):
    e = np.eye(x.size)
    return np.array([(fun(x + h * e[i]) - fun(x - h * e[i])) / (2 * h) for i in range(x.size)])


# --- quadratic forms -------------------------------------------------------------------


def test_quadratic_form_hand_value():
    # x^T W x = 2*1 + 1*4 = 6, L x = 1 - 2 = -1, c = 3
    W = np.array([[2.0, 0.0], [0.0, 1.0]])
    L = np.array([1.0, -1.0])
    assert quadratic_form(np.array([1.0, 2.0]), W, L, 3.0) == pytest.approx(8.0)


def test_quadratic_form_nonsymmetric_matches_symmetric_part():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3, 3))
    L = rng.standard_normal(3)
    x = rng.standard_normal(3)
    W_sym = (W + W.T) / 2
    assert quadratic_form(x, W, L, 0.5) == pytest.approx(quadratic_form(x, W_sym, L, 0.5))


@pytest.mark.parametrize("symmetric", [True, False])
def test_quadratic_gradient_matches_finite_differences(symmetric):
    rng = np.random.default_rng(1)
    W = random_spd(4, rng) if symmetric else rng.standard_normal((4, 4))
    L = rng.standard_normal(4)
    x = rng.standard_normal(4)
    g = quadratic_gradient(x, W, L)
    assert g.shape == (4,)
    expected = numerical_gradient(lambda z: quadratic_form(z, W, L, 0.0), x)
    np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-6)


def test_quadratic_minimizer_hand_value():
    # q(x) = 2 x1^2 + x2^2 - 4 x1 + 2 x2  ->  x* = (1, -1)
    W = np.array([[2.0, 0.0], [0.0, 1.0]])
    L = np.array([-4.0, 2.0])
    np.testing.assert_allclose(quadratic_minimizer(W, L), [1.0, -1.0])


def test_quadratic_minimizer_zero_gradient_and_minimum():
    rng = np.random.default_rng(2)
    W = random_spd(3, rng)
    L = rng.standard_normal(3)
    x_star = quadratic_minimizer(W, L)
    assert x_star.shape == (3,)
    np.testing.assert_allclose(quadratic_gradient(x_star, W, L), 0.0, atol=1e-10)
    q_star = quadratic_form(x_star, W, L, 0.0)
    for _ in range(20):
        x = x_star + rng.standard_normal(3)
        assert quadratic_form(x, W, L, 0.0) > q_star


@pytest.mark.parametrize(
    "W",
    [
        np.array([[1.0, 0.0], [0.0, -1.0]]),  # saddle
        np.array([[1.0, 0.0], [0.0, 0.0]]),  # flat direction
        np.array([[1.0, 4.0], [0.0, 1.0]]),  # nonsymmetric, indefinite symmetric part
    ],
)
def test_quadratic_minimizer_rejects_non_positive_definite(W):
    with pytest.raises(ValueError):
        quadratic_minimizer(W, np.zeros(2))


# --- regressor -------------------------------------------------------------------------


def test_polynomial_regressor_columns():
    t = np.array([-1.0, 0.0, 2.0])
    M = polynomial_regressor(t, 2)
    np.testing.assert_allclose(M, [[1.0, -1.0, 1.0], [0.0, 0.0, 1.0], [4.0, 2.0, 1.0]])


def test_polynomial_regressor_matches_polyval():
    t = np.linspace(-2, 3, 11)
    p = np.array([0.5, -1.0, 2.0, 3.0])
    np.testing.assert_allclose(polynomial_regressor(t, 3) @ p, np.polyval(p, t))


def test_polynomial_regressor_degree_zero():
    M = polynomial_regressor(np.array([1.0, 2.0, 3.0]), 0)
    np.testing.assert_allclose(M, np.ones((3, 1)))


# --- least squares ---------------------------------------------------------------------


def test_least_squares_hand_values():
    p_hat = least_squares(M_HAND, Y_HAND)
    assert p_hat.shape == (2,)
    np.testing.assert_allclose(p_hat, P_HAND)
    np.testing.assert_allclose(residuals(M_HAND, Y_HAND, p_hat), R_HAND, atol=1e-12)
    assert criterion(M_HAND, Y_HAND, p_hat) == pytest.approx(1 / 6)


def test_least_squares_regressor_of_hand_example():
    np.testing.assert_allclose(polynomial_regressor(T_HAND, 1), M_HAND)


def test_least_squares_recovers_exact_parameters():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((30, 4))
    p = rng.standard_normal(4)
    np.testing.assert_allclose(least_squares(M, M @ p), p, rtol=1e-10)


def test_least_squares_matches_lstsq():
    rng = np.random.default_rng(4)
    M = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    expected = np.linalg.lstsq(M, y, rcond=None)[0]
    np.testing.assert_allclose(least_squares(M, y), expected, rtol=1e-8)


def test_polynomial_fit_matches_polyfit():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 4, 25)
    y = rng.standard_normal(25)
    p_hat = least_squares(polynomial_regressor(t, 3), y)
    np.testing.assert_allclose(p_hat, np.polyfit(t, y, 3), rtol=1e-8)


def test_residuals_are_orthogonal_to_columns_of_M():
    rng = np.random.default_rng(6)
    M = rng.standard_normal((40, 3))
    y = rng.standard_normal(40)
    r = residuals(M, y, least_squares(M, y))
    assert r.shape == (40,)
    np.testing.assert_allclose(M.T @ r, 0.0, atol=1e-10)


def test_criterion_is_minimal_at_estimate():
    rng = np.random.default_rng(7)
    M = rng.standard_normal((20, 3))
    y = rng.standard_normal(20)
    p_hat = least_squares(M, y)
    j_hat = criterion(M, y, p_hat)
    assert j_hat == pytest.approx(np.sum(residuals(M, y, p_hat) ** 2))
    for _ in range(20):
        assert criterion(M, y, p_hat + 0.1 * rng.standard_normal(3)) > j_hat


def test_least_squares_is_minimizer_of_its_quadratic_form():
    # j(p) = p^T (M^T M) p - 2 y^T M p + y^T y
    rng = np.random.default_rng(8)
    M = rng.standard_normal((15, 3))
    y = rng.standard_normal(15)
    W, L = M.T @ M, -2 * M.T @ y
    np.testing.assert_allclose(least_squares(M, y), quadratic_minimizer(W, L), rtol=1e-10)
    p = rng.standard_normal(3)
    assert criterion(M, y, p) == pytest.approx(quadratic_form(p, W, L, y @ y))


# --- simulator -------------------------------------------------------------------------


def test_simulate_without_noise_is_exact():
    t = np.linspace(-1, 1, 9)
    p = np.array([1.0, -2.0, 0.5])
    y = simulate(p, t, 0.0, np.random.default_rng(9))
    assert y.shape == (9,)
    np.testing.assert_allclose(y, np.polyval(p, t))


def test_simulate_is_reproducible():
    t = np.linspace(0, 1, 5)
    p = np.array([1.0, 0.0])
    y1 = simulate(p, t, 0.3, np.random.default_rng(10))
    y2 = simulate(p, t, 0.3, np.random.default_rng(10))
    np.testing.assert_array_equal(y1, y2)


def test_simulate_noise_statistics():
    t = np.linspace(0, 1, 20_000)
    p = np.array([2.0, -1.0])
    sigma = 0.4
    noise = simulate(p, t, sigma, np.random.default_rng(11)) - np.polyval(p, t)
    assert noise.mean() == pytest.approx(0.0, abs=4 * sigma / np.sqrt(t.size))
    assert noise.std() == pytest.approx(sigma, rel=0.03)


# --- plotting helpers (scaffolding) ----------------------------------------------------


def test_plot_helpers_draw():
    W = np.array([[2.0, 0.5], [0.5, 1.0]])
    t = np.linspace(0, 1, 5)
    plot.plot_quadratic(W, np.zeros(2), 0.0, (-1, 1), (-1, 1), x_star=np.zeros(2))
    plot.plot_fit(t, t, t, t, y_true=t)
    plot.plot_residuals(t, t - 0.5)
    plt.close("all")
