"""Plotting helpers shared by the chapters.

These functions only draw; they never estimate anything. Each takes an optional ``ax`` and
returns the axes it drew on, so that figures can be composed in the notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def _axes(ax: Axes | None) -> Axes:
    return plt.figure().add_subplot() if ax is None else ax


def plot_quadratic(
    W: np.ndarray,
    L: np.ndarray,
    c: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    x_star: np.ndarray | None = None,
    levels: int = 20,
    ax: Axes | None = None,
) -> Axes:
    """Contour lines of ``q(x) = x^T W x + L x + c`` for ``x`` in the plane.

    The quadratic form is evaluated directly here, so the figure does not depend on the
    implementation being tested. ``x_star`` (optional) is marked with a cross.
    """
    ax = _axes(ax)
    x1, x2 = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
    X = np.stack([x1, x2], axis=-1)  # (200, 200, 2)
    q = np.einsum("...i,ij,...j->...", X, W, X) + X @ L + c
    cs = ax.contour(x1, x2, q, levels=levels, cmap="viridis")
    ax.clabel(cs, fontsize=7)
    if x_star is not None:
        ax.plot(*x_star, "rx", markersize=10, mew=2, label=r"$x^*$")
        ax.legend()
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    return ax


def plot_fit(
    t: np.ndarray,
    y: np.ndarray,
    t_model: np.ndarray,
    y_model: np.ndarray,
    y_true: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Measurements ``(t, y)`` as dots and a fitted curve ``(t_model, y_model)`` as a line.

    ``y_true`` (optional), sampled on ``t_model``, is drawn dashed.
    """
    ax = _axes(ax)
    ax.plot(t, y, "o", ms=4, label="measurements $y$")
    ax.plot(t_model, y_model, "-", label="fitted model")
    if y_true is not None:
        ax.plot(t_model, y_true, "k--", lw=1, label="ground truth")
    ax.set_xlabel("$t$")
    ax.legend()
    return ax


def plot_residuals(t: np.ndarray, r: np.ndarray, ax: Axes | None = None) -> Axes:
    """Residuals ``r`` against ``t`` as a stem plot."""
    ax = _axes(ax)
    ax.stem(t, r, basefmt="k-")
    ax.set_xlabel("$t$")
    ax.set_ylabel("residual $r$")
    return ax
