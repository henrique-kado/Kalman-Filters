---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 01 — Linear least squares

*Lecture C1.*

## Motivation

Every filter in this book answers the same question: given measurements that are imperfect,
which value of the unknown explains them best? Least squares is the oldest and simplest
answer. It also contains, in a static setting, most of the ideas the Kalman filter will
reuse: a criterion that is a quadratic form, a minimizer obtained by solving a linear
system, and residuals that tell how well the model fits.

This chapter covers three things:

1. quadratic forms and how to minimize them;
2. linear least squares, which turns out to be the minimization of a particular quadratic
   form;
3. the geometry of the solution: filtered measurements and residuals.

```{note}
**Notation.** The lectures write the matrix of a quadratic form as $Q$. This book calls it
$W$, because $Q$ is reserved for the covariance of the state noise from chapter 04 on.
See [Notation](notation.md).
```

## Theory

### Quadratic forms

A quadratic form on $\mathbb{R}^n$ is a function

$$
q(x) = x^\top W x + L x + c,
$$

where $W$ is an $n \times n$ matrix, $L$ a row vector of size $n$ and $c$ a scalar.

Only the symmetric part of $W$ matters. Because $x^\top W x$ is a scalar, it equals its own
transpose $x^\top W^\top x$, hence

$$
x^\top W x = x^\top \tfrac{W + W^\top}{2}\, x .
$$

From now on we may assume $W$ symmetric without loss of generality.

**Gradient.** Expanding $q(x + h)$ to first order in $h$ gives
$q(x + h) = q(x) + \big(x^\top (W + W^\top) + L\big) h + o(\lVert h \rVert)$, so the gradient
(as a column vector) is

$$
\nabla q(x) = (W + W^\top)\, x + L^\top \quad\overset{W \text{ symmetric}}{=}\quad 2 W x + L^\top .
$$

**Minimizer.** If $W$ is symmetric **positive definite**, $q$ is strictly convex and its
level sets are ellipsoids. It has a unique minimizer, where the gradient vanishes:

$$
2 W x^* + L^\top = 0 \quad\Longleftrightarrow\quad x^* = -\tfrac{1}{2} W^{-1} L^\top .
$$

If $W$ has a negative eigenvalue, $q$ is unbounded below (a saddle); if it has a zero
eigenvalue, the minimum, when it exists, is not unique. In both cases there is no single
$x^*$ to return.

In code, $x^*$ is obtained by *solving* the linear system $2 W x = -L^\top$, never by
computing $W^{-1}$ explicitly: solving is cheaper and more accurate.

### Linear least squares

We have $N$ measurements $y \in \mathbb{R}^N$ and a model that is **linear in the
parameters** $p \in \mathbb{R}^n$:

$$
y = M p + \beta ,
$$

where $M$ is a known $N \times n$ matrix (the *regressor*) and $\beta$ an unknown error.
The model may be very nonlinear in other variables. Fitting a parabola
$y_i = p_1 t_i^2 + p_2 t_i + p_3$ is a linear least-squares problem, with row $i$ of $M$
equal to $(t_i^2,\ t_i,\ 1)$.

The least-squares estimate minimizes the sum of squared errors

$$
j(p) = \lVert M p - y \rVert^2 .
$$

Expanding the norm shows that $j$ is a quadratic form in $p$:

$$
j(p) = p^\top \underbrace{M^\top M}_{W}\, p \;\underbrace{-\,2\, y^\top M}_{L}\, p
\;+\; \underbrace{y^\top y}_{c} .
$$

$M^\top M$ is always symmetric and positive semi-definite, since
$p^\top M^\top M p = \lVert M p \rVert^2 \ge 0$. It is positive *definite* exactly when
$M p = 0$ forces $p = 0$, that is, when the columns of $M$ are linearly independent. Under
that condition the minimizer formula of the previous section applies with
$W = M^\top M$ and $L^\top = -2 M^\top y$:

$$
\boxed{\hat{p} = (M^\top M)^{-1} M^\top y .}
$$

The equations $M^\top M \hat{p} = M^\top y$ are called the **normal equations**.

### Filtered measurements and residuals

Two vectors derived from $\hat{p}$ are useful:

- the **filtered measurement** $\hat{y} = M \hat{p}$, what the fitted model predicts;
- the **residual** $r = y - \hat{y}$, what it fails to explain.

The normal equations can be rewritten $M^\top (y - M\hat{p}) = 0$, that is

$$
M^\top r = 0 .
$$

The residual is orthogonal to every column of $M$. Geometrically, $\hat{y}$ is the
orthogonal projection of $y$ onto the subspace spanned by the columns of $M$, and $r$ is
the part of $y$ that no choice of parameters can reach. Small, structureless residuals
indicate a good model; residuals with a visible pattern (a trend, a curvature) indicate
that the model is missing something.

## Simulator

The data in this chapter come from a polynomial with known coefficients $p$, measured at
times $t_i$ with independent Gaussian errors:

$$
y_i = p_1 t_i^d + \dots + p_d\, t_i + p_{d+1} + \beta_i, \qquad \beta_i \sim \mathcal{N}(0, \sigma^2).
$$

Implement `simulate` in `kalman/models/polynomial.py`.

## Estimator

Implement, in `kalman/least_squares.py`:

| Function | Returns |
|---|---|
| `quadratic_form(x, W, L, c)` | $q(x)$ |
| `quadratic_gradient(x, W, L)` | $\nabla q(x)$, for any square $W$ |
| `quadratic_minimizer(W, L)` | $x^*$; `ValueError` if $W$ is not positive definite |
| `polynomial_regressor(t, degree)` | $M$, columns $t^d, \dots, t, 1$ |
| `least_squares(M, y)` | $\hat{p}$, from the normal equations |
| `residuals(M, y, p)` | $r = y - M p$ |
| `criterion(M, y, p)` | $j(p)$ |

They are done when `uv run pytest tests/test_ch01.py` passes.

## Experiments

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

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

rng = np.random.default_rng(2026)
```

### A quadratic form and its minimizer

The level sets of a positive definite quadratic form are ellipses centred on $x^*$; the
eigenvectors of $W$ give their axes.

```{code-cell} python
W = np.array([[2.0, 0.8], [0.8, 1.0]])
L = np.array([-2.0, 1.0])
c = 1.0

x_star = quadratic_minimizer(W, L)
print("x*        =", x_star)
print("q(x*)     =", quadratic_form(x_star, W, L, c))
print("grad q(x*) =", quadratic_gradient(x_star, W, L))

plot.plot_quadratic(W, L, c, xlim=(-2, 3), ylim=(-3, 2), x_star=x_star)
plt.title("Level sets of $q$");
```

### Fitting a parabola

Twenty noisy measurements of a parabola, and the least-squares fit.

```{code-cell} python
p_true = np.array([0.5, -2.0, 1.0])
sigma = 0.5
t = np.linspace(0.0, 5.0, 20)
y = simulate(p_true, t, sigma, rng)

M = polynomial_regressor(t, 2)
p_hat = least_squares(M, y)
r = residuals(M, y, p_hat)

print("p_true =", p_true)
print("p_hat  =", p_hat.round(3))
print("j(p_hat) =", round(criterion(M, y, p_hat), 3))
print("M^T r =", (M.T @ r).round(12))

t_fine = np.linspace(0.0, 5.0, 200)
M_fine = polynomial_regressor(t_fine, 2)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
plot.plot_fit(t, y, t_fine, M_fine @ p_hat, y_true=M_fine @ p_true, ax=ax1)
plot.plot_residuals(t, r, ax=ax2)
fig.tight_layout()
```

The residuals scatter around zero with no visible pattern, and the product $M^\top r$ is zero
up to rounding: the residual is orthogonal to the columns of $M$.

### Fitting the wrong model

Fitting a straight line to the same data leaves the curvature in the residuals.

```{code-cell} python
M1 = polynomial_regressor(t, 1)
p1 = least_squares(M1, y)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
plot.plot_fit(t, y, t_fine, polynomial_regressor(t_fine, 1) @ p1, y_true=M_fine @ p_true, ax=ax1)
plot.plot_residuals(t, residuals(M1, y, p1), ax=ax2)
ax2.set_title("Residuals of a straight-line fit")
fig.tight_layout()
```

### More parameters are not always better

With only ten measurements, fit polynomials of increasing degree. The criterion
$j(\hat{p})$ on the fitting data can only decrease when the degree grows, because each model
contains the previous one. The error on *new* measurements of the same parabola, averaged
over many fresh data sets, tells a different story.

```{code-cell} python
t10 = np.linspace(0.0, 5.0, 10)
y10 = simulate(p_true, t10, sigma, rng)
y_new = [simulate(p_true, t10, sigma, rng) for _ in range(200)]

degrees = range(0, 9)
j_fit, j_new = [], []
for d in degrees:
    Md = polynomial_regressor(t10, d)
    pd = least_squares(Md, y10)
    j_fit.append(criterion(Md, y10, pd))
    j_new.append(np.mean([criterion(Md, yn, pd) for yn in y_new]))

fig, ax = plt.subplots()
ax.semilogy(degrees, j_fit, "o-", label="data used for the fit")
ax.semilogy(degrees, j_new, "s-", label="new measurements (mean)")
ax.axvline(2, color="k", ls=":", lw=1)
ax.set_xlabel("polynomial degree $d$")
ax.set_ylabel(r"$j(\hat{p})$")
ax.legend();
```

Beyond the true degree (dotted line), the extra parameters fit the noise of this particular
data set and generalize worse.

### How much does $\hat{p}$ vary?

Repeat the experiment many times with fresh noise and look at the spread of $\hat{p}$. If the
errors are independent with variance $\sigma^2$, linear algebra (chapter 03) predicts that
the covariance of $\hat{p}$ is $\sigma^2 (M^\top M)^{-1}$.

```{code-cell} python
runs = 2000
p_hats = np.array([least_squares(M, simulate(p_true, t, sigma, rng)) for _ in range(runs)])

print("mean of p_hat      :", p_hats.mean(axis=0).round(3), "   p_true:", p_true)
print("sample std of p_hat:", p_hats.std(axis=0).round(4))
print("predicted std      :", np.sqrt(np.diag(sigma**2 * np.linalg.inv(M.T @ M))).round(4))

fig, ax = plt.subplots()
ax.plot(p_hats[:, 0], p_hats[:, 1], ".", ms=2, alpha=0.4)
ax.plot(*p_true[:2], "rx", ms=10, mew=2, label="true $(p_1, p_2)$")
ax.set_xlabel(r"$\hat{p}_1$")
ax.set_ylabel(r"$\hat{p}_2$")
ax.legend();
```

The estimates are centred on the truth, and the cloud is an ellipse: $\hat{p}_1$ and
$\hat{p}_2$ are strongly correlated. Chapter 03 makes this precise with covariance matrices
and confidence ellipses.

## Exercises

1. Show that the minimum value of a positive definite quadratic form is
   $q(x^*) = c - \tfrac{1}{4} L W^{-1} L^\top$. Check it numerically on the first experiment.
2. With $N$ measurements, what is $j(\hat{p})$ when the degree is $N - 1$? What goes wrong
   when it is $N$ or more? Try it, and explain with the rank of $M$.
3. *Conditioning.* For $t \in [0, 10]$ and degree 8, compare `np.linalg.cond(M)` with
   `np.linalg.cond(M.T @ M)`. Compare your `least_squares` with `np.linalg.lstsq`, which
   avoids the normal equations. Then rescale $t$ to $[-1, 1]$ and repeat.
4. *Weighted least squares.* If measurement $i$ has variance $\sigma_i^2$, it is natural to
   minimize $(M p - y)^\top R^{-1} (M p - y)$ with $R = \operatorname{diag}(\sigma_i^2)$.
   Write it as a quadratic form and derive $\hat{p}$. This criterion comes back in the Kalman
   filter.
5. *Linear in the parameters.* Fit $y = a \sin(\omega t) + b \cos(\omega t) + c$ for a known
   $\omega$. Build $M$ by hand and reuse `least_squares`. Why does the same approach fail when
   $\omega$ is also unknown? (Chapter 02.)

## References

- ENSTA course *Kalman*, lecture C1.
- L. Jaulin, *kalmooc* (Kalman filtering course and exercises).
- G. Strang, *Introduction to Linear Algebra*, chapter on orthogonality: projections and
  least-squares approximations.
