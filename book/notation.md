# Notation

This book keeps the symbols of the ENSTA *Kalman* lectures, with three exceptions chosen to
match the wider literature: the state covariance is written $P$ and the two noise
covariances $Q$ and $R$.

## Renamed symbols

| Meaning                              | This book | Lectures      |
|--------------------------------------|-----------|---------------|
| Covariance of the state estimate     | $P$       | $\Gamma$      |
| Covariance of the state noise        | $Q$       | $\Gamma_\alpha$ |
| Covariance of the measurement noise  | $R$       | $\Gamma_\beta$  |
| Matrix of a quadratic form (ch. 01)  | $W$       | $Q$           |

The last line avoids a clash: in the lectures, $Q$ first appears as the matrix of a quadratic
form $x^\top Q x + L x + c$. Here that matrix is called $W$, so that $Q$ always means the state
noise covariance.

## Models

A linear discrete-time system with state $x_k \in \mathbb{R}^n$ and measurement
$y_k \in \mathbb{R}^m$:

$$
\begin{aligned}
x_{k+1} &= A_k x_k + u_k + \alpha_k, & \alpha_k &\sim \mathcal{N}(0, Q_k) \\
y_k     &= C_k x_k + \beta_k,         & \beta_k  &\sim \mathcal{N}(0, R_k)
\end{aligned}
$$

where $u_k$ is a known input. Nonlinear systems replace the two equations by
$x_{k+1} = f(x_k, u_k) + \alpha_k$ and $y_k = g(x_k) + \beta_k$.

## Estimates

| Symbol | Meaning | In code |
|---|---|---|
| $\hat{x}_{k\vert k-1},\ P_{k\vert k-1}$ | prediction of $x_k$ from measurements up to $k-1$ | `x_pred`, `P_pred` |
| $\hat{x}_{k\vert k},\ P_{k\vert k}$ | correction of $x_k$ using $y_k$ as well | `x_corr`, `P_corr` |
| $\hat{x}_{k\vert N},\ P_{k\vert N}$ | smoothed estimate using all $N$ measurements (ch. 09) | `x_smooth`, `P_smooth` |
| $\hat{x}_{0\vert -1},\ P_{0\vert -1}$ | initial guess, before any measurement | `x0`, `P0` |
| $\tilde{y}_k$ | innovation | `y_tilde` |
| $S_k$ | covariance of the innovation | `S` |
| $K_k$ | Kalman gain | `K` |

## One step of the filter

As in the lectures, a step is a **corrector followed by a predictor**. Starting from
$\hat{x}_{0|-1}$ and $P_{0|-1}$, for each $k$:

$$
\begin{aligned}
\tilde{y}_k &= y_k - C_k \hat{x}_{k|k-1} \\
S_k &= C_k P_{k|k-1} C_k^\top + R_k \\
K_k &= P_{k|k-1} C_k^\top S_k^{-1} \\
\hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k \tilde{y}_k, \qquad P_{k|k} = (I - K_k C_k)\, P_{k|k-1} \\
\hat{x}_{k+1|k} &= A_k \hat{x}_{k|k} + u_k, \qquad P_{k+1|k} = A_k P_{k|k} A_k^\top + Q_k
\end{aligned}
$$

For nonlinear systems (extended Kalman filter), $A_k$ and $C_k$ become Jacobians:
$A_k = \partial f / \partial x$ evaluated at $\hat{x}_{k|k}$, and
$C_k = \partial g / \partial x$ evaluated at $\hat{x}_{k|k-1}$.

## Other recurring objects

- **Optimal linear estimator** (ch. 03): $\hat{x} = \bar{x} + K (y - \bar{y})$ with
  $K = P_{xy} P_y^{-1}$.
- **Confidence ellipse** at probability $\eta$ (ch. 03): the set of $x$ such that
  $(x - \bar{x})^\top P^{-1} (x - \bar{x}) \le -2 \ln(1 - \eta)$.
- **Least squares** (ch. 01–02): parameters $p$, data $y$, model $f(p)$ (or $M p$ when
  linear), criterion $j(p) = \lVert f(p) - y \rVert^2$; in the linear case
  $\hat{p} = (M^\top M)^{-1} M^\top y$.

## Code conventions

- Vectors are 1-D NumPy arrays of shape `(n,)`; matrices have shape `(n, n)` or `(m, n)`.
  There are no column vectors.
- When several measurements arrive at the same instant, their rows of $C$ are stacked.
- A missing measurement is `NaN`. Missing entries are dropped together with the matching
  rows of $C$ and of $R$; if all are missing, the correction step is skipped.
- Every random draw goes through an explicit `rng: numpy.random.Generator` argument, so that
  every figure in this book is reproducible.
