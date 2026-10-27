# Week 2 — Random Variables and Joint Distributions

Lecture: Lecture 2 — Review of probability theory (handout §2.1–2.5). Problem sheet: Tutorial 2 (keep a local copy in [`../course-material/`](../course-material/); it is not published here).

## Files

| File | What it contains |
|---|---|
| [`01_Concepts.md`](01_Concepts.md) · [PDF](pdf/01_Concepts.pdf) | Joint/marginal/conditional pmfs (hat example), correlation and covariance, correlation coefficient and Cauchy–Schwarz, LMMSE derivation, conditional expectation and tower property, uncorrelated vs independent, cdf/pdf (wheel and model-train examples), joint pdfs, integrating over regions, hierarchical models, mixed RVs and the Dirac delta, Gaussian pdf/cdf/Q/erf, central limit theorem, bivariate Gaussian and contours, random vectors and covariance matrices, multivariate Gaussian (with the bivariate derivation), recipes, cheat sheet |
| [`02_Solutions.md`](02_Solutions.md) · [PDF](pdf/02_Solutions.pdf) | Tutorial 2, Problems 1–3, fully worked |
| [`python/tutorial2_verify.py`](python/tutorial2_verify.py) | Exact joint table and moments for the cards, symbolic integration for Problems 2 and 3, Monte Carlo for Problem 2; figures in [`python/figures/`](python/figures/) |

Run with `python python/tutorial2_verify.py` (needs `numpy sympy matplotlib`).

## Progress

- [x] Lecture 2 concepts
- [x] Problem 1 — Two discrete random variables (cards)
- [x] Problem 2 — Bivariate pdf from marginal and conditional
- [x] Problem 3 — Uniform pdf on a triangle

## Key results

- **Problem 1:** $\mu_X = 8$, $\mu_Y = 2$, $\sigma_X^2 = 3$, $\sigma_Y^2 = 1$, $E\lbrace XY\rbrace = 16$, $\sigma_{XY} = \rho_{XY} = 0$ — uncorrelated but **not** independent; $E\lbrace Y\mid X = x\rbrace = 1,2,2,3,2,2,1$ for $x = 5,\dots,11$.
- **Problem 2:** $f_{XY} = 1/x^3$ on $x\ge1$, $0\le y\le x$; $f_Y(y) = \frac12$ on $[0,1]$, $\frac{1}{2y^2}$ for $y>1$.
- **Problem 3:** $a = \frac12$; $Y\mid X = x\sim U[\frac x2, 2-\frac x2]$; $X\mid Y = y\sim U[0, 2\min(y,2-y)]$; $\rho_{XY} = 0$ (dependent but uncorrelated).

![Joint pmf of the card problem](python/figures/p1_joint_pmf.png)
