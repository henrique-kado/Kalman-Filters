# Kalman-Filters — Design Spec

- **Date:** 2026-11-07
- **Repo:** `henrique-kado/Kalman-Filters`
- **Status:** approved in brainstorming, pending written-spec review

## 1. Purpose

A study + portfolio repository for learning Kalman filtering **from scratch in Python**, aligned with the ENSTA course *Kalman* (lectures C1–C6, based on L. Jaulin's *kalmooc*). The owner implements every estimator, filter and simulator himself, on top of prepared scaffolding (theory drafts, skeletons, tests) and with PR review. The result is published as a Jupyter Book on GitHub Pages and presented through a README with animations.

**Success criteria**

- Every chapter: tests pass, the chapter notebook executes end to end in CI, key figures are generated.
- From chapter 04 on: filters are statistically consistent (NEES/NIS inside χ² bounds on seeded Monte-Carlo runs).
- The published book and README let a visitor understand each filter without opening the code.

## 2. Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Goal | Learn the theory from scratch; implement by hand with NumPy |
| Scope | Full track: foundations → KF → EKF → UKF → particle filter → RTS → Q/R tuning → information filter → IMM |
| Format | Python package `kalman/` + one notebook per chapter importing it |
| Who implements | Owner implements estimators, filters **and simulators**; scaffolding covers skeletons, tests, plotting/animation helpers, CI and book config; every chapter is reviewed in a PR |
| Language | Repository text in **English** |
| Notation | Lecture notation, except **P** (state covariance), **Q** (state noise), **R** (measurement noise) |
| Publication | Jupyter Book 2 (MyST) on GitHub Pages + README with a few GIFs |
| Lecture PDFs | Never committed (third-party material); referenced by name only |

## 3. Notation

Lecture symbols are kept; three are renamed. Every chapter restates the mapping once.

| Concept | Repo | Lectures | Code |
|---|---|---|---|
| Evolution model | x_{k+1} = A_k x_k + u_k + α_k | same | `A`, `u` |
| Measurement model | y_k = C_k x_k + β_k | same | `C`, `y` |
| State-noise covariance | **Q** | Γ_α | `Q` |
| Measurement-noise covariance | **R** | Γ_β | `R` |
| Predicted estimate / covariance | x̂_{k\|k-1}, **P**_{k\|k-1} | x̂_{k\|k-1}, Γ_{k\|k-1} | `x_pred`, `P_pred` |
| Corrected estimate / covariance | x̂_{k\|k}, **P**_{k\|k} | x̂_{k\|k}, Γ_{k\|k} | `x_corr`, `P_corr` |
| Innovation, its covariance, gain | ỹ_k, S_k, K_k | same | `y_tilde`, `S`, `K` |
| Nonlinear models | f(x, u), g(x) | same | `f`, `g` |
| Jacobians | A_k = ∂f/∂x at x̂_{k\|k}; C_k = ∂g/∂x at x̂_{k\|k-1} | same | `jac_f`, `jac_g` |
| Initialization | x̂_{0\|-1}, P_{0\|-1} | x̂_{0\|-1}, Γ_{0\|-1} | `x0`, `P0` |
| Smoothed estimate (ch. 09) | x̂_{k\|N}, P_{k\|N} | — | `x_smooth`, `P_smooth` |
| Static linear estimator (C3) | x̂ = x̄ + K(y − ȳ), K = P_xy P_y⁻¹ | Γ_xy Γ_y⁻¹ | — |
| Confidence ellipse (C3) | (x − x̄)ᵀ P⁻¹ (x − x̄) ≤ −2 ln(1 − η) | same with Γ | — |
| Least squares (C1–C2) | p, y, M, f(p), j(p) = ‖f(p) − y‖², p̂ = (MᵀM)⁻¹Mᵀy | same | — |
| Quadratic form (C1) | xᵀ**W**x + Lx + c | xᵀQx + Lx + c | `W` |

The C1 quadratic-form matrix is renamed **W** so that Q has a single meaning; chapter 01 notes this.

**Cycle order.** As in the lectures, one filter step is **corrector then predictor**, starting from x̂_{0|-1}, P_{0|-1}:

1. ỹ_k = y_k − C_k x̂_{k|k-1}
2. S_k = C_k P_{k|k-1} C_kᵀ + R_k
3. K_k = P_{k|k-1} C_kᵀ S_k⁻¹
4. x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k; P_{k|k} = (I − K_k C_k) P_{k|k-1}
5. x̂_{k+1|k} = A_k x̂_{k|k} + u_k; P_{k+1|k} = A_k P_{k|k} A_kᵀ + Q_k

## 4. Learning track

| # | Chapter | Lecture | Module(s) | Example problems |
|---|---|---|---|---|
| 01 | Linear least squares | C1 | `least_squares.py` | polynomial fit, quadratic forms, residuals |
| 02 | Nonlinear estimation: Gauss–Newton, Monte-Carlo, simulated annealing | C2 | `least_squares.py` | exponential model, local vs global minima |
| 03 | Covariance, confidence ellipse, optimal linear estimator | C3 | `gaussian.py` | robot position cloud, motor, trochoid |
| 04 | Discrete Kalman filter | C4 | `kf.py` | thermometer, motor |
| 05 | Localization with walls and dead reckoning | C5 | `kf.py` | robot with known heading, walls n·p = ρ |
| 06 | Extended Kalman filter | C6 | `ekf.py` | robot with unknown heading + beacon range, inverted pendulum, two radars |
| 07 | Unscented Kalman filter | — | `ukf.py` | same problems as 06, side-by-side comparison |
| 08 | Particle filter | — | `particle.py` | single-beacon ring ambiguity (non-Gaussian posterior) |
| 09 | RTS smoother | — | `smoother.py` | smoothing chapter-05 trajectories |
| 10 | Tuning Q and R (NEES, NIS, Joseph form) | — | `metrics.py` | deliberately mistuned filters |
| 11 | Information filter | — | `information.py` | multi-sensor fusion |
| 12 | IMM | — | `imm.py` | target switching between straight motion and turning |

Chapters 07–12 use outside references but are written entirely in the notation above.

## 5. Repository architecture

```
Kalman-Filters/
├── kalman/                  # package
│   ├── __init__.py
│   ├── least_squares.py     # owner
│   ├── gaussian.py          # owner
│   ├── kf.py  ekf.py  ukf.py  particle.py  smoother.py  information.py  imm.py   # owner
│   ├── metrics.py           # owner (RMSE, NEES, NIS)
│   ├── models/              # owner: simulators with known ground truth
│   │   └── thermometer.py  robot.py  pendulum.py  radar.py  ...
│   └── plot.py              # scaffolding: plotting + animation helpers
├── tests/                   # scaffolding: test_ch01.py ... test_ch12.py
├── book/                    # Jupyter Book 2
│   ├── myst.yml
│   ├── index.md
│   ├── notation.md          # notation + mapping to lectures (own words, no slide copies)
│   └── ch01-least-squares.md ... ch12-imm.md   # MyST Markdown notebooks
├── scripts/make_animations.py   # scaffolding
├── assets/                  # committed GIFs for README (< 2 MB each)
├── docs/design/             # design docs
├── .github/workflows/tests.yml  book.yml
├── .python-version          # 3.14 (read by uv)
├── .gitattributes           # * text=auto eol=lf
├── pyproject.toml
├── uv.lock                  # committed: identical environments on Ubuntu and Windows
└── README.md
```

## 6. API

Pure functions, kalmooc style. Each function does one step and maps one-to-one to the lecture equations.

```python
# kf.py
def kalman_correc(x_pred, P_pred, y, R, C):
    """Return x_corr, P_corr, y_tilde, S. NaN entries of y are dropped together
    with the matching rows of C and rows/cols of R (wall not visible, C5).
    If every entry is NaN, skip correction: return x_pred, P_pred, None, None."""


def kalman_predict(x_corr, P_corr, u, Q, A):
    """Return x_pred, P_pred for k+1."""


def kalman(x_pred, P_pred, u, y, Q, R, A, C):
    """One full cycle (corrector then predictor).
    Return x_pred_next, P_pred_next, x_corr, P_corr."""


# ekf.py
def ekf_correc(x_pred, P_pred, y, R, g, jac_g): ...
def ekf_predict(x_corr, P_corr, u, Q, f, jac_f): ...
```

UKF, particle filter, smoother, information filter and IMM follow the same pattern (`*_correc` / `*_predict` plus a full-step helper). Their exact signatures are fixed in each chapter's skeleton.

**Conventions**

- Vectors are 1-D arrays `(n,)`, matrices are `(n, n)`. No column vectors.
- Several measurements at the same instant: stack the rows of C (C5).
- Covariance update: P = (I − KC)P as in the lectures. The Joseph form is an exercise in chapter 10.
- Randomness only through an explicit `rng: np.random.Generator` argument. No global seeding.

**Simulators**

```python
def simulate(..., N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return x_true (N, n) and y (N, m); rows of y are NaN when no measurement is available."""
```

## 7. Chapter workflow

1. **Scaffolding** is prepared on branch `chXX-<name>`:
   - the chapter notebook (theory, lecture references such as "see C4, *Le filtre complet*", experiment cells);
   - module skeletons with signatures and docstrings, raising `NotImplementedError`;
   - the chapter tests.
2. **The owner implements** the simulator, then the filter, until `pytest tests/test_chXX.py` passes, then runs the notebook experiments.
3. **Review** of the Pull Request (math, clarity, fidelity to lectures). The owner merges.

**Definition of done:** tests green, notebook executes in CI, key figures produced.

**Notebook template:** Motivation → Theory (with notation mapping) → Simulator → Filter → Experiments → Consistency (NEES/NIS, from ch. 04) → Exercises → References.

## 8. Testing strategy

1. **Hand-computed values from the lectures** (reference vectors below).
2. **Properties:** output shapes; P symmetric and positive semi-definite; no correction when y is all-NaN, partial correction when only some entries are NaN; P_{k|k} ≤ P_{k|k-1} on the diagonal for linear-Gaussian cases.
3. **Simulator statistics:** the sample covariance of generated noise matches Q and R within tolerance (seeded).
4. **Consistency (from ch. 04):** on seeded Monte-Carlo runs, the average NEES lies inside the two-sided 95 % χ² interval.
5. **Cross-checks:**
   - chapter 03: static estimator = least squares with prior;
   - chapter 07: UKF = KF on linear models.

**Reference vectors (from the lectures)**

- **C3, propagation.** x̄ = (2.0, 1.0), P_x = [[0.08, 0.036], [0.036, 0.02]], A = [[1, 0.2], [0, 1]], b = (1.0, 0.3), Q = 0.01·I → ȳ = (3.2, 1.3).
- **C3, static estimator.** x = (1, 2, 3, 4), y = (3, 5, 7, 9), with population (co)variances → K = 0.5, x̂ = 0.5y − 0.5.
- **C4, thermometer.**
  - Correction: x̂_{k|k-1} = 6.7, y_k = 7.0 → ỹ_k = 0.3.
  - Prediction: P_{k|k} = 0.25, Q = 0.09, A = 1 → P_{k+1|k} = 0.34.
- **C5, dead reckoning.** x̂ = (2, 2, 1), θ = 0, a = 0, Δt = 1, P = diag(0.16, 0.16, 0.04), Q = diag(0.04, 0.04, 0.01) → x̂_{k+1|k} = (3, 2, 1), diag P = (0.24, 0.20, 0.05), P[p_x, v] = 0.04.
- **C5, wall correction.** From the prediction above: y = (3.28, 1.76), C = [[1, 0, 0], [0, 1, 0]], R = 0.04·I → K = [[6/7, 0], [0, 5/6], [1/7, 0]], x̂_{k|k} = (3.24, 1.80, 1.04).
- **C6, beacon range.** b = (1, 1), p̂ = (4, 5) → ŷ = 5, C = [0.6, 0.8, 0, 0]; y = 5.5 → ỹ = 0.5.

## 9. Environment and cross-platform support

The project must work identically on **Ubuntu** and **Windows**.

- **uv** (Astral) manages the interpreter, the virtual environment and the lock file, with the same commands on both systems:
  - `uv sync --all-extras` installs everything;
  - `uv run pytest` runs the tests;
  - `uv run jupyter book build --execute` builds the book (run from `book/`);
  - `uv run python scripts/make_animations.py` generates the GIFs.
- **Python 3.14** (latest bugfix series with a mature scientific ecosystem), pinned in `.python-version` and installed by uv itself. Python 3.15.0 (released 2026-10-09) can be adopted once NumPy, SciPy, Matplotlib and Jupyter Book publish wheels and the CI matrix passes on it.
- `requires-python = ">=3.12"` so the package stays installable elsewhere.
- **No Makefile.** `make` is not available on Windows; the commands above are documented in the README.
- **No shell scripts.** All automation is Python. Paths use `pathlib`; files are opened with `encoding="utf-8"`.
- **`.gitattributes`** normalizes line endings to LF.
- **Node.js LTS** is required by Jupyter Book 2 (MyST) for local book builds; CI installs it automatically.
- **CI matrix:** `tests.yml` runs on `ubuntu-latest` and `windows-latest`.
- **Repository location on Windows:** outside OneDrive (e.g. `C:\Users\henri\Projects\Kalman-Filters`). Syncing `.git` and `.venv` through OneDrive causes file locks and conflicts.

## 10. Tooling

- **Jupytext** with MyST Markdown notebooks (`.md`) as the only committed notebook format: clean diffs, no outputs in git. `.ipynb` files are gitignored.
- **pytest** with `addopts = "-q --tb=short"`.
- **Ruff** (lint + format) via **pre-commit**.
- **`book/notation.md`** is the single summary of the lectures.

## 11. Publication

- **`tests.yml`:** Ruff + pytest on every push and PR, on Ubuntu and Windows.
- **`book.yml`:** on push to `main`, build the book with execution and deploy to GitHub Pages at `henrique-kado.github.io/Kalman-Filters`. One-time setting: *Settings → Pages → Source: GitHub Actions*.
- **Runtime budget:** each chapter executes in under ~1 minute (sized Monte-Carlo, fixed seeds).
- **README:**
  - badges, a one-paragraph pitch, a hero GIF (chapter 06 EKF robot with confidence ellipse);
  - a track table (chapter, lecture, status ✅/🚧, book link);
  - quick start: install uv, then `uv sync --all-extras` and `uv run pytest`.
- **Animations:** `scripts/make_animations.py` writes GIFs for chapters 05, 06, 08 and 12 to `assets/`.
- **Dependencies:** Python 3.14 for development (≥ 3.12 supported); numpy, scipy, matplotlib; extras `[dev]` (pytest, ruff, jupytext, pre-commit) and `[book]` (jupyter-book).

## 12. Out of scope

- Using filtering libraries (FilterPy etc.) inside `kalman/`. They may appear only in an optional comparison cell.
- Real-sensor datasets. All data is simulated with known ground truth.
- Continuous-time Kalman–Bucy filter, SLAM.
- Publishing the lecture PDFs or reproducing slide content.

## 13. First milestone

**Chapter 00 — repository setup** (scaffolding):
- pyproject + uv lock, `.python-version`, `.gitattributes`, CI (Ubuntu + Windows), Jupyter Book skeleton with `index.md` and `notation.md`;
- pre-commit, README skeleton.

Then chapter 01 follows the workflow in §7.
