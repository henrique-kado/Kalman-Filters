# Kalman Filters from Scratch

A study notebook on state estimation: every estimator and filter in this book is
implemented by hand in Python with NumPy, tested, and run on simulated systems where the
ground truth is known. The path goes from linear least squares to the Kalman filter and on
to its nonlinear and multiple-model extensions.

The notation follows the ENSTA course *Kalman* (lectures C1–C6), itself based on
L. Jaulin's *kalmooc*, with three renamed symbols. See [Notation](notation.md) before
reading any chapter.

## Learning track

| #  | Chapter                                                          | Lecture | Status  |
|----|------------------------------------------------------------------|---------|---------|
| 01 | [Linear least squares](ch01-least-squares.md)                    | C1      | done    |
| 02 | Nonlinear estimation: Gauss–Newton, Monte-Carlo, simulated annealing | C2  | planned |
| 03 | Covariance, confidence ellipse, optimal linear estimator         | C3      | planned |
| 04 | Discrete Kalman filter                                           | C4      | planned |
| 05 | Localization with walls and dead reckoning                       | C5      | planned |
| 06 | Extended Kalman filter                                           | C6      | planned |
| 07 | Unscented Kalman filter                                          | —       | planned |
| 08 | Particle filter                                                  | —       | planned |
| 09 | Rauch–Tung–Striebel smoother                                     | —       | planned |
| 10 | Tuning Q and R (NEES, NIS, Joseph form)                          | —       | planned |
| 11 | Information filter                                               | —       | planned |
| 12 | Interacting multiple models (IMM)                                | —       | planned |

Chapters 07–12 go beyond the course and draw on other references, written in the same
notation.

## How each chapter is built

1. Theory, in the notation of this book, with references to the matching lecture.
2. A simulator of the system, with known ground truth.
3. The filter, implemented in the `kalman` package and covered by tests.
4. Experiments: estimates against the truth, confidence ellipses, error statistics.
5. From chapter 04 on, a consistency check (NEES/NIS): does the filter's covariance match
   its actual errors?
