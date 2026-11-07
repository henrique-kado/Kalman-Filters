# Kalman Filters from Scratch

[![tests](https://github.com/henrique-kado/Kalman-Filters/actions/workflows/tests.yml/badge.svg)](https://github.com/henrique-kado/Kalman-Filters/actions/workflows/tests.yml)
[![book](https://github.com/henrique-kado/Kalman-Filters/actions/workflows/book.yml/badge.svg)](https://henrique-kado.github.io/Kalman-Filters/)

A study of state estimation, from linear least squares to the Kalman filter and its
nonlinear and multiple-model extensions. Every estimator and filter is implemented by hand
in Python with NumPy, covered by tests, and run on simulated systems where the ground truth
is known, so that each estimate can be compared with reality and each covariance checked
for consistency.

📖 **Read the book:** <https://henrique-kado.github.io/Kalman-Filters/>

## Learning track

| #  | Chapter                                                     | Lecture | Status |
|----|-------------------------------------------------------------|---------|--------|
| 01 | Linear least squares                                        | C1      | 🚧     |
| 02 | Nonlinear estimation: Gauss–Newton, Monte-Carlo, annealing  | C2      | 🚧     |
| 03 | Covariance, confidence ellipse, optimal linear estimator    | C3      | 🚧     |
| 04 | Discrete Kalman filter                                      | C4      | 🚧     |
| 05 | Localization with walls and dead reckoning                  | C5      | 🚧     |
| 06 | Extended Kalman filter                                      | C6      | 🚧     |
| 07 | Unscented Kalman filter                                     | —       | 🚧     |
| 08 | Particle filter                                             | —       | 🚧     |
| 09 | Rauch–Tung–Striebel smoother                                | —       | 🚧     |
| 10 | Tuning Q and R (NEES, NIS, Joseph form)                     | —       | 🚧     |
| 11 | Information filter                                          | —       | 🚧     |
| 12 | Interacting multiple models (IMM)                           | —       | 🚧     |

## Repository layout

```
kalman/     filters, estimators and simulators (the package)
tests/      tests for each chapter
book/       Jupyter Book: one MyST Markdown notebook per chapter
docs/       design notes
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/); it installs the right
Python version by itself. Works the same on Linux and Windows.

```bash
uv sync --all-extras    # create the environment
uv run pytest           # run the tests
```

To build the book locally (needs [Node.js](https://nodejs.org/) LTS):

```bash
cd book
uv run jupyter book start    # live preview at http://localhost:3000
```

## Acknowledgements

The notation and the first six chapters follow the *Kalman* course taught at ENSTA, based
on L. Jaulin's *kalmooc*. The text and code in this repository are original; the course
material itself is not reproduced here.

## License

[MIT](LICENSE)
