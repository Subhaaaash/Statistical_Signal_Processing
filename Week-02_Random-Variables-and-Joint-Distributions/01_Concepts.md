---
title: "Week 2 — Concepts"
subtitle: "Statistical Signal Processing · Lecture 2: Review of Probability Theory (joint, continuous and Gaussian RVs, random vectors)"
---

# Week 2 — Concepts

**Course:** Statistical Signal Processing (L.048.24014, Universität Paderborn, Signal and System Theory Group, Dr.-Ing. Tanuj Hasija), WiSe 2026/27
**Lecture:** 2 — Review of probability theory (handout §2.1–2.5)
**Covers:** everything needed for Tutorial 2 (Problems 1–3: two discrete RVs, bivariate pdfs from a marginal and a conditional, uniform pdf on a triangle) and the rest of Lecture 2 (Gaussian distribution, central limit theorem, random vectors, multivariate Gaussian)

The basics of Lecture 2 (axioms, conditional probability, Bayes, one discrete RV, expectation, variance) are in the [Week 1 concepts](../Week-01_Introduction-and-Probability-Basics/01_Concepts.md) and are only recalled briefly here. This sheet builds the step from **one** random variable to **two or more**, and from **discrete** to **continuous**.

The worked solutions are in [`02_Solutions.md`](02_Solutions.md). The script [`python/tutorial2_verify.py`](python/tutorial2_verify.py) checks every number (exact fractions, symbolic integration with SymPy and Monte Carlo).

**Notation** (as in the lecture):

| Symbol | Meaning |
|---|---|
| $P_{XY}(x,y)$ | joint pmf of discrete $X, Y$ |
| $F_X(x)$, $f_X(x)$ | cdf and pdf of a continuous RV |
| $F_{XY}(x,y)$, $f_{XY}(x,y)$ | joint cdf and joint pdf |
| $f_{X\mid Y}(x\mid y)$ | conditional pdf of $X$ given $Y = y$ |
| $\mu_X$, $\sigma_X^2$ | mean and variance |
| $\mu_{XY} = E\lbrace XY\rbrace$ | **correlation** of $X$ and $Y$ |
| $\sigma_{XY}$ | **covariance** of $X$ and $Y$ |
| $\rho_{XY}$ | correlation **coefficient** |
| $\mathbf{X}$, $\boldsymbol{\mu}_{\mathbf{X}}$, $\mathbf{R}_{\mathbf{XX}}$ | random vector, mean vector, covariance matrix |
| $\mathcal{N}(\mu,\sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |

Careful: in this course "correlation" $\mu_{XY} = E\lbrace XY\rbrace$ is **not** the correlation coefficient $\rho_{XY}$. Tutorial 2 asks for both.

---

## Contents

1. [Recap from Week 1](#1-recap-from-week-1)
2. [Two discrete random variables](#2-two-discrete-random-variables)
3. [Expectation with two RVs: correlation and covariance](#3-expectation-with-two-rvs-correlation-and-covariance)
4. [Correlation coefficient and linear estimation](#4-correlation-coefficient-and-linear-estimation)
5. [Conditional expectation](#5-conditional-expectation)
6. [Uncorrelated versus independent](#6-uncorrelated-versus-independent)
7. [Continuous random variables: cdf and pdf](#7-continuous-random-variables-cdf-and-pdf)
8. [Two continuous random variables](#8-two-continuous-random-variables)
9. [Integrating over regions — the key skill for Tutorial 2](#9-integrating-over-regions--the-key-skill-for-tutorial-2)
10. [Mixed random variables and the Dirac delta](#10-mixed-random-variables-and-the-dirac-delta)
11. [The Gaussian distribution](#11-the-gaussian-distribution)
12. [Central limit theorem](#12-central-limit-theorem)
13. [The bivariate Gaussian](#13-the-bivariate-gaussian)
14. [Random vectors, mean vectors and covariance matrices](#14-random-vectors-mean-vectors-and-covariance-matrices)
15. [The multivariate Gaussian](#15-the-multivariate-gaussian)
16. [Problem-solving recipes for Tutorial 2](#16-problem-solving-recipes-for-tutorial-2)
17. [Cheat sheet](#17-cheat-sheet)

---

## 1. Recap from Week 1

- Conditional probability $P(A\mid B) = P(AB)/P(B)$; independence $P(AB) = P(A)P(B)$; total probability and Bayes.
- A discrete RV has a pmf $P_X(x)\ge0$, $\sum_xP_X(x) = 1$.
- $E\lbrace g(X)\rbrace = \sum_xg(x)P_X(x)$ is linear; $\sigma_X^2 = E\lbrace X^2\rbrace - \mu_X^2$; $\mathrm{var}(aX+b) = a^2\sigma_X^2$.
- Counting: $\binom{n}{k}$ unordered selections; with equally likely outcomes $P(E) = |E|/|\Omega|$.

---

## 2. Two discrete random variables

### 2.1 Joint, marginal and conditional distributions

Two RVs $X$ and $Y$ defined on the **same** experiment are described together by the

- **joint probability distribution** $P_{XY}(x,y)\triangleq P(X = x, Y = y)$ — the probability that both values occur together. It is best written as a **table** with one row per $x$ and one column per $y$; all entries are $\ge0$ and sum to 1.

From the joint table everything else follows:

- **Marginal distributions** (sum out the other variable — row and column sums of the table):
$$P_X(x) = \sum_yP_{XY}(x,y),\qquad P_Y(y) = \sum_xP_{XY}(x,y).$$
This is the law of total probability: the events $\lbrace Y = y\rbrace$ partition $\Omega$.
- **Conditional distributions** (one row or column of the table, renormalised):
$$P_{X\mid Y}(x\mid y) = \frac{P_{XY}(x,y)}{P_Y(y)},\qquad P_{Y\mid X}(y\mid x) = \frac{P_{XY}(x,y)}{P_X(x)}.$$
For each fixed condition, the conditional distribution is a proper pmf (sums to 1 over the free variable).

The marginals do **not** determine the joint distribution (many tables have the same row and column sums). The joint distribution contains the extra information about how $X$ and $Y$ depend on each other.

### 2.2 Independence

$X$ and $Y$ are (statistically) **independent** if
$$P_{XY}(x,y) = P_X(x)\,P_Y(y)\qquad\text{for all }x, y.$$
Then $P_{X\mid Y}(x\mid y) = P_X(x)$ and $P_{Y\mid X}(y\mid x) = P_Y(y)$. In table form: every row is proportional to every other row.

**Quick test for dependence:** find a **zero** in the joint table whose row and column marginals are both positive. Then $P_{XY}(x,y) = 0\ne P_X(x)P_Y(y)$, so they are dependent. One counterexample suffices; to prove independence you must check **all** cells.

### 2.3 Lecture example — the hat with four balls

Balls "AA", "AB", "BA", "BB"; draw two. $X = 1$ if "AA" is among them, else 0; $Y$ = number of "B"s on the two balls. The $\binom42 = 6$ pairs are equally likely:

| pair | AA,AB | AA,BA | AA,BB | AB,BA | AB,BB | BA,BB |
|---|---|---|---|---|---|---|
| $(x,y)$ | (1,1) | (1,1) | (1,2) | (0,2) | (0,3) | (0,3) |

| $P_{XY}$ | $y=1$ | $y=2$ | $y=3$ | $P_X(x)$ |
|---|---|---|---|---|
| $x=0$ | 0 | 1/6 | 1/3 | 1/2 |
| $x=1$ | 1/3 | 1/6 | 0 | 1/2 |
| $P_Y(y)$ | 1/3 | 1/3 | 1/3 | |

Conditional $P_{X\mid Y}$: divide each column by its sum (1/3): column $y=1$ → $(0, 1)$, $y=2$ → $(1/2, 1/2)$, $y=3$ → $(1, 0)$. Not independent: $P_{XY}(0,1) = 0\ne\frac12\cdot\frac13$.

---

## 3. Expectation with two RVs: correlation and covariance

For a function $g(X,Y)$:
$$E\lbrace g(X,Y)\rbrace = \sum_x\sum_yg(x,y)\,P_{XY}(x,y)\quad\text{(discrete)},$$
$$E\lbrace g(X,Y)\rbrace = \int\!\!\int g(x,y)\,f_{XY}(x,y)\,dx\,dy\quad\text{(continuous)}.$$
To compute $E\lbrace X\rbrace$ you may use either the joint table ($g = x$) or the marginal — they give the same number.

Two special choices:

- **Correlation** ($g = xy$): $\mu_{XY} = E\lbrace XY\rbrace$.
- **Covariance** ($g = (x-\mu_X)(y-\mu_Y)$): $\sigma_{XY} = E\lbrace(X-\mu_X)(Y-\mu_Y)\rbrace$.

**Shortcut** (expand and use linearity, like the variance shortcut):
$$\sigma_{XY} = E\lbrace XY\rbrace - \mu_XE\lbrace Y\rbrace - \mu_YE\lbrace X\rbrace + \mu_X\mu_Y = \boxed{\mu_{XY} - \mu_X\mu_Y}$$

Meaning: $\sigma_{XY}>0$ if $X$ and $Y$ tend to be above (or below) their means **together**; $\sigma_{XY}<0$ if one tends to be above when the other is below. Note $\sigma_{XX} = \sigma_X^2$.

Useful rules:

- $\mathrm{cov}(aX + b, cY + d) = ac\,\sigma_{XY}$ (shifts drop out).
- $\mathrm{var}(X + Y) = \sigma_X^2 + \sigma_Y^2 + 2\sigma_{XY}$.
- $X, Y$ are **uncorrelated** if $E\lbrace XY\rbrace = E\lbrace X\rbrace E\lbrace Y\rbrace$, i.e. $\sigma_{XY} = 0$.
- **Independent ⇒ uncorrelated** (because then $E\lbrace XY\rbrace = \sum\sum xyP_X(x)P_Y(y) = \mu_X\mu_Y$). The converse is false (§6).

---

## 4. Correlation coefficient and linear estimation

### 4.1 Cauchy–Schwarz and the correlation coefficient

**Cauchy–Schwarz inequality:** $|\sigma_{XY}|\le\sigma_X\sigma_Y$, with equality iff $X = aY + b$ (a linear relation, almost surely).

*Proof.* For every real $t$, $0\le\mathrm{var}(Y - tX) = \sigma_Y^2 - 2t\sigma_{XY} + t^2\sigma_X^2$. A quadratic in $t$ that is never negative has a non-positive discriminant: $4\sigma_{XY}^2 - 4\sigma_X^2\sigma_Y^2\le0$. Equality means $\mathrm{var}(Y - tX) = 0$ for some $t$, i.e. $Y - tX$ is a constant.

Hence the **correlation coefficient**
$$\rho_{XY} = \frac{\sigma_{XY}}{\sigma_X\sigma_Y}\in[-1, 1]$$
is a normalised (unit-free) covariance:

- $|\rho_{XY}| = 1$: exact linear relation, $X$ is perfectly linearly predictable from $Y$.
- $\rho_{XY} = 0$: uncorrelated; a **linear** estimator of one from the other is useless.
- It does not change under $X\to aX + b$ with $a>0$ (sign flips for $a<0$).

### 4.2 Best linear estimator (LMMSE) — derivation

Predict $Y$ from $X$ with $\hat Y = aX + b$, minimising the mean-squared error
$$\text{MSE}(a,b) = E\lbrace(Y - aX - b)^2\rbrace = E\lbrace Y^2\rbrace - 2aE\lbrace XY\rbrace - 2b\mu_Y + 2ab\mu_X + a^2E\lbrace X^2\rbrace + b^2.$$
Set the partial derivatives to zero:
$$\frac{\partial\,\text{MSE}}{\partial b} = -2\mu_Y + 2a\mu_X + 2b = 0\;\Rightarrow\; b = \mu_Y - a\mu_X,$$
$$\frac{\partial\,\text{MSE}}{\partial a} = -2E\lbrace XY\rbrace + 2b\mu_X + 2aE\lbrace X^2\rbrace = 0.$$
Substitute $b$ into the second equation: $-E\lbrace XY\rbrace + \mu_X\mu_Y - a\mu_X^2 + aE\lbrace X^2\rbrace = 0$, i.e. $-\sigma_{XY} + a\sigma_X^2 = 0$:
$$a = \frac{\sigma_{XY}}{\sigma_X^2} = \rho_{XY}\frac{\sigma_Y}{\sigma_X},\qquad b = \mu_Y - \rho_{XY}\frac{\sigma_Y}{\sigma_X}\mu_X,$$
$$\boxed{\hat Y = \mu_Y + \rho_{XY}\frac{\sigma_Y}{\sigma_X}(X - \mu_X)}$$
The minimum MSE: with $\tilde X = X - \mu_X$, $\tilde Y = Y - \mu_Y$, the error is $\tilde Y - a\tilde X$, so
$$\text{MSE} = \sigma_Y^2 - 2a\sigma_{XY} + a^2\sigma_X^2 = \sigma_Y^2 - \frac{\sigma_{XY}^2}{\sigma_X^2} = \boxed{\sigma_Y^2(1 - \rho_{XY}^2)}$$
So $|\rho| = 1$ ⇒ MSE = 0; $\rho = 0$ ⇒ MSE $= \sigma_Y^2$ and the best linear estimator is just the constant $\hat Y = \mu_Y$. This formula is the seed of the LMMSE estimator and the Wiener filter later in the course.

---

## 5. Conditional expectation

**Conditioning on an event** $\lbrace X = x\rbrace$: the mean of the conditional distribution,
$$E\lbrace Y\mid X = x\rbrace = \sum_yy\,P_{Y\mid X}(y\mid x)\quad\text{or}\quad\int y\,f_{Y\mid X}(y\mid x)\,dy.$$
This is a **number** for each $x$, i.e. a function $h(x)$.

**Conditioning on a random variable:** $E\lbrace Y\mid X\rbrace = h(X)$ is itself a **random variable** (a function of $X$).

**Law of total expectation (tower property):**
$$E\lbrace Y\rbrace = E\lbrace E\lbrace Y\mid X\rbrace\rbrace = \sum_xE\lbrace Y\mid X = x\rbrace P_X(x).$$
(Inner expectation over $Y$, outer over $X$.) Derivation, as on the slide: $\sum_yyP_Y(y) = \sum_yy\sum_xP_{Y\mid X}(y\mid x)P_X(x) = \sum_x\big[\sum_yyP_{Y\mid X}(y\mid x)\big]P_X(x)$.

A useful generalisation: $E\lbrace g(X)\,Y\rbrace = E\lbrace g(X)\,E\lbrace Y\mid X\rbrace\rbrace$ ("take out what is known"). With $g(X) = X$:
$$E\lbrace XY\rbrace = E\lbrace X\,E\lbrace Y\mid X\rbrace\rbrace.$$
**Consequence:** if $E\lbrace Y\mid X = x\rbrace = c$ is the same for every $x$, then $E\lbrace XY\rbrace = c\,\mu_X$ and $\mu_Y = c$, so $\sigma_{XY} = 0$: $X$ and $Y$ are uncorrelated. This gives a very fast answer in Tutorial 2, Problem 3.

$E\lbrace Y\mid X\rbrace$ is also the best (minimum MSE) estimator of $Y$ from $X$ among **all** functions, not just linear ones (Chapter 5 of the course).

---

## 6. Uncorrelated versus independent

| | Independent | Uncorrelated |
|---|---|---|
| Definition | joint = product of marginals (whole distribution) | $E\lbrace XY\rbrace = \mu_X\mu_Y$ (one number) |
| Says | knowing $X$ tells you **nothing** about $Y$ | knowing $X$ does not help a **linear** prediction of $Y$ |
| Implication | independent ⇒ uncorrelated | uncorrelated ⇏ independent |

Classic counterexample: $X$ uniform on $\lbrace-1,0,1\rbrace$, $Y = X^2$. Then $E\lbrace XY\rbrace = E\lbrace X^3\rbrace = 0 = \mu_X\mu_Y$, so uncorrelated, but $Y$ is a function of $X$ — totally dependent. The dependence is **non-linear**, which $\rho$ cannot see.

Both problems in Tutorial 2 with a covariance turn out to be of this type (dependent but uncorrelated).

**Exception:** for **jointly Gaussian** RVs, uncorrelated ⇔ independent (§13).

---

## 7. Continuous random variables: cdf and pdf

### 7.1 Why probabilities of single points vanish

**Lecture example (spinning wheel):** the stopping angle $X$ is equally likely anywhere in $[0,2\pi)$, so $P(x_1<X\le x_2) = \frac{x_2-x_1}{2\pi}$. Letting the interval shrink to a point gives $P(X = x) = 0$ for every $x$. For continuous RVs we therefore work with probabilities of **intervals**, via the cdf.

### 7.2 Cumulative distribution function and probability density function

$$F_X(x)\triangleq P(X\le x) = \int_{-\infty}^{x}f_X(\xi)\,d\xi,\qquad f_X(x) = \frac{d}{dx}F_X(x).$$
Properties:

- $P(a<X\le b) = F_X(b) - F_X(a) = \int_a^bf_X(x)\,dx$ (area under the pdf).
- $F_X$ is non-decreasing, $F_X(-\infty) = 0$, $F_X(\infty) = 1$.
- $f_X(x)\ge0$ and $\int_{-\infty}^{\infty}f_X(x)\,dx = 1$, **but $f_X(x)$ can exceed 1** — it is a density (probability per unit length), not a probability. Intuition: $P(x<X\le x + dx)\approx f_X(x)\,dx$.
- For continuous RVs, $<$ and $\le$ give the same probabilities.

Spinning wheel: $f_X(x) = \frac{1}{2\pi}$ on $[0,2\pi)$, 0 elsewhere — the **uniform** distribution. In general, $X\sim U[a,b]$ has $f_X(x) = \frac{1}{b-a}$ on $[a,b]$, mean $\frac{a+b}{2}$, variance $\frac{(b-a)^2}{12}$.

### 7.3 Lecture example — the model train (finding a pdf via the cdf)

A train runs at constant speed on a circle of radius $r$; at a random time its angle $\Theta$ is uniform on $[-\pi,\pi)$. Find the pdf of its x-position $X = r\cos\Theta$.

1. **Easy ranges:** $x<-r$ ⇒ $F_X(x) = 0$; $x\ge r$ ⇒ $F_X(x) = 1$.
2. **Translate the event:** for $-r\le x\le r$, $\lbrace X\le x\rbrace = \lbrace\cos\Theta\le x/r\rbrace = \lbrace\arccos(x/r)\le|\Theta|\le\pi\rbrace$ (two arcs, symmetric).
3. **Probability:** $F_X(x) = 2\cdot\frac{\pi - \arccos(x/r)}{2\pi} = \frac1\pi\arccos(-x/r)$ (using $\pi - \arccos u = \arccos(-u)$).
4. **Differentiate:** $f_X(x) = \frac{1}{\pi\sqrt{r^2 - x^2}}$ for $-r<x<r$, 0 elsewhere.

The density is large near $\pm r$: the train spends more time there in terms of x-position (it moves mostly vertically). The **cdf method** — express $\lbrace g(\Theta)\le x\rbrace$ as an event about $\Theta$, compute its probability, differentiate — works for any transformation $X = g(\Theta)$.

### 7.4 Expectations for continuous RVs

Replace sums by integrals and the pmf by the pdf:
$$E\lbrace g(X)\rbrace = \int_{-\infty}^{\infty}g(x)f_X(x)\,dx,\qquad \mu_X = \int_{-\infty}^{\infty}xf_X(x)\,dx,$$
$$\sigma_X^2 = \int_{-\infty}^{\infty}(x-\mu_X)^2f_X(x)\,dx = E\lbrace X^2\rbrace - \mu_X^2.$$
All rules (linearity, variance shortcut, covariance shortcut, tower property) carry over unchanged.

**Moments may not exist.** If the integral diverges, the moment is undefined. Example (Tutorial 2, Problem 2): $f_X(x) = 1/x^2$ for $x\ge1$ is a valid pdf, but $E\lbrace X\rbrace = \int_1^\infty\frac{x}{x^2}dx = \int_1^\infty\frac1x\,dx = \infty$. Such "heavy-tailed" densities are perfectly legitimate.

---

## 8. Two continuous random variables

### 8.1 Definitions

| Object | Definition |
|---|---|
| joint cdf | $F_{XY}(x,y) = P(X\le x, Y\le y)$ |
| joint pdf | $f_{XY}(x,y) = \frac{\partial^2}{\partial x\,\partial y}F_{XY}(x,y)$ |
| probability of a region $D$ | $P((X,Y)\in D) = \iint_Df_{XY}(x,y)\,dx\,dy$ (volume under the surface) |
| normalisation | $\iint_{\mathbb{R}^2}f_{XY}\,dx\,dy = 1$, $f_{XY}\ge0$ |
| marginal pdfs | $f_X(x) = \int_{-\infty}^{\infty}f_{XY}(x,y)\,dy$, $f_Y(y) = \int_{-\infty}^{\infty}f_{XY}(x,y)\,dx$ |
| conditional pdfs | $f_{X\mid Y}(x\mid y) = \frac{f_{XY}(x,y)}{f_Y(y)}$, $f_{Y\mid X}(y\mid x) = \frac{f_{XY}(x,y)}{f_X(x)}$ |
| independence | $F_{XY} = F_XF_Y$ ⇔ $f_{XY}(x,y) = f_X(x)f_Y(y)$ |

### 8.2 Reading the definitions geometrically

- The **support** is the set where $f_{XY}>0$. Always sketch it first.
- **Marginal** $f_X(x)$: fix $x$, integrate the joint pdf along the vertical line through $x$ (a "projection" onto the x-axis).
- **Conditional** $f_{Y\mid X}(y\mid x)$: take the **slice** of the joint pdf along the vertical line at $x$, and divide by its area $f_X(x)$ so that it integrates to 1 in $y$. It is defined only where $f_X(x)>0$.
- **Uniform joint pdf on a region $R$** ($f_{XY} = a$ on $R$): $a = 1/\text{area}(R)$; every slice is flat, so **every conditional pdf is uniform on the slice**, and $f_X(x) = a\times$(length of the vertical slice at $x$).
- **Independence needs a "rectangular" support.** If $f_{XY} = f_Xf_Y$, the support is the product of the two marginal supports (a rectangle or product set). A triangle or a region like $\lbrace0\le y\le x\rbrace$ can **never** support independent RVs: the set of possible $y$ values depends on $x$.

### 8.3 Building a joint pdf from a marginal and a conditional (hierarchical model)

Rearranging the definition of the conditional pdf:
$$\boxed{f_{XY}(x,y) = f_{Y\mid X}(y\mid x)\,f_X(x)}$$
This is the continuous chain rule. It is how two-stage experiments are modelled: "first draw $X$, then, given $X = x$, draw $Y$ from some distribution depending on $x$". Tutorial 2, Problem 2 is exactly this. Then the marginal of $Y$ is $f_Y(y) = \int f_{Y\mid X}(y\mid x)f_X(x)\,dx$ (continuous total probability), and Bayes for densities gives $f_{X\mid Y}(x\mid y) = \frac{f_{Y\mid X}(y\mid x)f_X(x)}{f_Y(y)}$.

**Watch the support:** "$Y$ uniform on $[0,x]$" means $f_{Y\mid X}(y\mid x) = \frac1x$ only for $0\le y\le x$ and **zero otherwise**. The zero part is what creates the $y$-dependent integration limits later.

---

## 9. Integrating over regions — the key skill for Tutorial 2

Most errors in joint-pdf problems are wrong integration limits. A systematic procedure:

1. **Sketch the support** and label its boundary curves with equations, e.g. $y = x/2$, $y = 2 - x/2$, $x = 0$.
2. **Choose an order.** To integrate over $y$ first (for $f_X$, or for an expectation), draw a **vertical** line at a fixed $x$ and read off where it enters and leaves the region: $y_{\text{low}}(x)\le y\le y_{\text{high}}(x)$. Then $x$ runs over the full x-range of the region.
3. **Split where the boundary changes.** If the entry or exit curve changes as the line moves (a corner of the region), split the outer range at the corner and use different limits on each part. For the triangle of Problem 3, horizontal lines exit through $x = 2y$ for $y\le1$ and through $x = 4 - 2y$ for $y\ge1$.
4. **Write the result with its range.** A marginal or conditional pdf is incomplete without "for … ≤ x ≤ …, 0 otherwise".
5. **Check that it integrates to 1.**

Tools that save time:

- **Area by geometry:** for a uniform pdf, $\iint_R a\,dx\,dy = a\cdot\text{area}(R)$; use triangle/rectangle formulas instead of integrals.
- **Centroid:** for a uniform pdf on a triangle with vertices $(x_i,y_i)$, $E\lbrace X\rbrace = \frac{x_1+x_2+x_3}{3}$, $E\lbrace Y\rbrace = \frac{y_1+y_2+y_3}{3}$.
- **Symmetry:** if the region is symmetric about the line $y = c$ (and the pdf is too), then $E\lbrace Y\mid X = x\rbrace = c$ for every $x$, hence $\sigma_{XY} = 0$ (§5).
- **Iterated expectation:** $E\lbrace XY\rbrace = \int x\,E\lbrace Y\mid X = x\rbrace f_X(x)\,dx$ often avoids a double integral.
- **Change of order:** $\int_1^\infty\!\int_0^x(\dots)\,dy\,dx = \int_0^\infty\!\int_{\max(1,y)}^\infty(\dots)\,dx\,dy$ — the inner limits in the second form come from a horizontal line at height $y$.

---

## 10. Mixed random variables and the Dirac delta

Some RVs have both point masses and a continuous part. **Lecture example:** $X = -1$ or $X = 1$ with probability $1/4$ each, and uniform on $(-\frac12,\frac12)$ with probability $1/2$. The cdf has jumps of $1/4$ at $\pm1$ and a ramp of total height $1/2$ on $(-\frac12,\frac12)$. Its generalised derivative is
$$f_X(x) = \tfrac14\delta(x-1) + \tfrac14\delta(x+1) + \tfrac12\,\mathrm{rect}(x),$$
where $\mathrm{rect}(x) = 1$ on $(-\frac12,\frac12)$ and the Dirac delta is defined by the sifting property $\int\delta(x-x_0)g(x)\,dx = g(x_0)$. With deltas, sums and integrals are treated uniformly, e.g. here $E\lbrace X^2\rbrace = \frac14 + \frac14 + \frac12\cdot\frac{1}{12} = \frac{13}{24}$. A discrete pmf can always be written as a sum of weighted deltas.

---

## 11. The Gaussian distribution

### 11.1 Gaussian pdf

$$X\sim\mathcal{N}(\mu_X,\sigma_X^2):\qquad f_X(x) = \frac{1}{\sqrt{2\pi}\,\sigma_X}\exp\Big\lbrace-\frac12\Big(\frac{x-\mu_X}{\sigma_X}\Big)^2\Big\rbrace.$$
Bell-shaped, symmetric about $\mu_X$, width set by $\sigma_X$ (peak height $\frac{1}{\sqrt{2\pi}\sigma_X}$, so smaller $\sigma$ ⇒ taller, narrower). **It is completely determined by its mean and variance.**

### 11.2 Gaussian cdf, $\Phi$, erf and Q

There is no closed form for the Gaussian cdf. Everything is reduced to the **standard normal** $\mathcal{N}(0,1)$ cdf
$$\Phi(z) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z}e^{-s^2/2}\,ds,\qquad F_X(x) = \Phi\Big(\frac{x-\mu_X}{\sigma_X}\Big)$$
(substitute $s = (\xi-\mu_X)/\sigma_X$, i.e. **standardise**: $Z = (X-\mu_X)/\sigma_X\sim\mathcal{N}(0,1)$).

Related functions:

| Function | Definition | Relation |
|---|---|---|
| error function | $\mathrm{erf}(z) = \frac{2}{\sqrt\pi}\int_0^ze^{-s^2}ds$ | |
| complementary error function | $\mathrm{erfc}(z) = 1 - \mathrm{erf}(z)$ | $\mathrm{erfc}(z) = 2\Phi(-\sqrt2z)$ |
| Q-function (tail probability) | $Q(z) = 1 - \Phi(z) = P(Z>z)$ | $Q(z) = \frac12\mathrm{erfc}(z/\sqrt2)$ |

Facts: $\Phi(-z) = 1 - \Phi(z)$; $\Phi(0) = 0.5$, $\Phi(1)\approx0.841$, $\Phi(2)\approx0.977$, $\Phi(3)\approx0.9987$. So $P(|X-\mu|\le\sigma)\approx0.683$, $P(|X-\mu|\le2\sigma)\approx0.954$, $P(|X-\mu|\le3\sigma)\approx0.997$. The Q-function is the standard way to write error probabilities of detectors (Chapter 6).

### 11.3 Why the Gaussian is everywhere (summary of properties)

1. Completely determined by mean and variance.
2. Jointly Gaussian + uncorrelated ⇒ independent.
3. Jointly Gaussian ⇒ marginals Gaussian (converse **not** true).
4. Jointly Gaussian ⇒ conditionals Gaussian.
5. $X$ Gaussian ⇒ $aX + b\sim\mathcal{N}(a\mu_X + b, a^2\sigma_X^2)$.
6. Independent Gaussians $X_n\sim\mathcal{N}(\mu_n,\sigma_n^2)$ ⇒ $\sum_nX_n\sim\mathcal{N}(\sum\mu_n,\sum\sigma_n^2)$ (means add, variances add — the "add" part holds for any independent RVs; staying Gaussian is special).
7. Central limit theorem: sums of many i.i.d. RVs tend to Gaussian (§12).

---

## 12. Central limit theorem

Let $X_1,\dots,X_N$ be **i.i.d.** (independent and identically distributed) with mean $\mu$ and variance $\sigma^2$. Define
$$Z = \frac{1}{\sqrt N}\sum_{n=1}^{N}(X_n - \mu).$$
Then $\mu_Z = 0$ and $\sigma_Z^2 = \frac1N\cdot N\sigma^2 = \sigma^2$ (variances of independent terms add), and **as $N\to\infty$, $Z$ approaches $\mathcal{N}(0,\sigma^2)$** — whatever the distribution of the $X_n$, even discrete. The lecture illustrates this with the sum of 1, 2, 3, 5, 10 dice: the flat distribution of one die becomes a bell after a few dice.

The $\sqrt N$ is the right scaling: dividing by $N$ (the sample mean) would make the variance go to zero; not dividing would make it grow. Practical reading: a sum of many small independent effects (thermal noise from many electrons, measurement errors) is approximately Gaussian. This justifies the Gaussian noise model used throughout SSP.

---

## 13. The bivariate Gaussian

Jointly Gaussian $X, Y$ have the pdf
$$f_{XY}(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho_{XY}^2}}\exp\left\lbrace-\frac{q(x,y)}{2(1-\rho_{XY}^2)}\right\rbrace,$$
$$q(x,y) = \Big(\frac{x-\mu_X}{\sigma_X}\Big)^2 - 2\rho_{XY}\frac{(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \Big(\frac{y-\mu_Y}{\sigma_Y}\Big)^2,$$
with five parameters $\mu_X, \mu_Y, \sigma_X, \sigma_Y, \rho_{XY}$.

- **Marginals** are Gaussian: $f_X(x) = \int f_{XY}\,dy = \mathcal{N}(\mu_X,\sigma_X^2)$ (complete the square in $y$).
- **Uncorrelated ⇒ independent:** with $\rho_{XY} = 0$ the cross term vanishes and the exponential splits into a product $f_X(x)f_Y(y)$.
- **Conditional** $Y\mid X = x$ is Gaussian with mean $\mu_Y + \rho_{XY}\frac{\sigma_Y}{\sigma_X}(x - \mu_X)$ and variance $\sigma_Y^2(1-\rho_{XY}^2)$ — exactly the linear estimator and its MSE from §4.2. **For jointly Gaussian RVs the best estimator is linear.**

**Level contours** (curves of constant pdf, take $\mu_X = \mu_Y = 0$): $q(x,y)$ equals a constant, so contours are ellipses.

- $\sigma_X = \sigma_Y$, $\rho = 0$: circles.
- $\rho = 0$, $\sigma_X\ne\sigma_Y$: $(x/\sigma_X)^2 + (y/\sigma_Y)^2 = c^2$, axis-aligned ellipses, long axis along the variable with the larger $\sigma$.
- $\sigma_X = \sigma_Y = \sigma$, $\rho\ne0$: $x^2 - 2\rho xy + y^2 = c^2\sigma^2(1-\rho^2)$, ellipses tilted at $45^\circ$ ($\rho>0$) or $135^\circ$ ($\rho<0$).
- General: arbitrary orientation (set by the eigenvectors of the covariance matrix, §14).

---

## 14. Random vectors, mean vectors and covariance matrices

Stack $n$ RVs in a column vector $\mathbf{X} = [X_1,\dots,X_n]^T$.

- **cdf** $F_{\mathbf{X}}(\mathbf{x}) = P(X_1\le x_1,\dots,X_n\le x_n)$ (the joint cdf); **pdf** $f_{\mathbf{X}}(\mathbf{x}) = \frac{\partial^n}{\partial x_1\cdots\partial x_n}F_{\mathbf{X}}(\mathbf{x})$. Similarly for two vectors $\mathbf{X}$ ($n$) and $\mathbf{Y}$ ($m$): the joint cdf/pdf of all $n+m$ components.
- $E\lbrace\cdot\rbrace$ of a vector or matrix acts **element-wise**.
- **Mean vector** $\boldsymbol{\mu}_{\mathbf{X}} = E\lbrace\mathbf{X}\rbrace$.
- **Covariance matrix**
$$\mathbf{R}_{\mathbf{XX}} = E\lbrace(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})^T\rbrace = E\lbrace\mathbf{XX}^T\rbrace - \boldsymbol{\mu}_{\mathbf{X}}\boldsymbol{\mu}_{\mathbf{X}}^T.$$
Its $(i,j)$ element is $\sigma_{X_iX_j}$: variances on the diagonal, covariances off the diagonal. For $n = 2$:
$$\mathbf{R}_{\mathbf{XX}} = \begin{bmatrix}\sigma_X^2 & \sigma_{XY}\\ \sigma_{XY} & \sigma_Y^2\end{bmatrix} = \begin{bmatrix}\sigma_X^2 & \rho\sigma_X\sigma_Y\\ \rho\sigma_X\sigma_Y & \sigma_Y^2\end{bmatrix}.$$
- **Correlation matrix** $E\lbrace\mathbf{XX}^T\rbrace$. **Cross-covariance** $\mathbf{R}_{\mathbf{XY}} = E\lbrace(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})(\mathbf{Y}-\boldsymbol{\mu}_{\mathbf{Y}})^T\rbrace$ (generally not symmetric, not even square).
- $\mathbf{X}$ and $\mathbf{Y}$ are **uncorrelated** if $E\lbrace\mathbf{XY}^T\rbrace = E\lbrace\mathbf{X}\rbrace E\lbrace\mathbf{Y}^T\rbrace$ (i.e. $\mathbf{R}_{\mathbf{XY}} = \mathbf{0}$), and **orthogonal** if $E\lbrace\mathbf{XY}^T\rbrace = \mathbf{0}$. Uncorrelated does not imply independent.

**Covariance (and correlation) matrices are symmetric and positive semidefinite.** Symmetric because $\sigma_{X_iX_j} = \sigma_{X_jX_i}$. PSD because for any fixed vector $\mathbf{z}$ the scalar $\mathbf{z}^T(\mathbf{X}-\boldsymbol{\mu})$ has non-negative second moment:
$$0\le E\lbrace[\mathbf{z}^T(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})]^2\rbrace = \mathbf{z}^TE\lbrace(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})^T\rbrace\mathbf{z} = \mathbf{z}^T\mathbf{R}_{\mathbf{XX}}\mathbf{z}.$$
($\mathbf{z}^T\mathbf{R}\mathbf{z}$ is the variance of the linear combination $\mathbf{z}^T\mathbf{X}$.) For $n = 2$, PSD is equivalent to $\sigma_X^2\ge0$, $\sigma_Y^2\ge0$ and $\det\mathbf{R} = \sigma_X^2\sigma_Y^2(1-\rho^2)\ge0$, i.e. $|\rho|\le1$ — Cauchy–Schwarz again.

**Linear transformation of any random vector:** if $\mathbf{Y} = \mathbf{AX} + \mathbf{b}$ then
$$\boldsymbol{\mu}_{\mathbf{Y}} = \mathbf{A}\boldsymbol{\mu}_{\mathbf{X}} + \mathbf{b},\qquad \mathbf{R}_{\mathbf{YY}} = \mathbf{A}\mathbf{R}_{\mathbf{XX}}\mathbf{A}^T$$
(since $\mathbf{Y}-\boldsymbol{\mu}_{\mathbf{Y}} = \mathbf{A}(\mathbf{X}-\boldsymbol{\mu}_{\mathbf{X}})$).

---

## 15. The multivariate Gaussian

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}\sqrt{\det\mathbf{R}_{\mathbf{XX}}}}\exp\Big\lbrace-\frac12(\mathbf{x}-\boldsymbol{\mu}_{\mathbf{X}})^T\mathbf{R}_{\mathbf{XX}}^{-1}(\mathbf{x}-\boldsymbol{\mu}_{\mathbf{X}})\Big\rbrace$$
(requires an invertible covariance matrix). Mean vector and covariance matrix determine it completely.

**Behaviour under linear transformation:** if $\mathbf{A}$ is $m\times n$ with rank $m$, then $\mathbf{Y} = \mathbf{AX}$ is $m$-dimensional Gaussian with $\boldsymbol{\mu}_{\mathbf{Y}} = \mathbf{A}\boldsymbol{\mu}_{\mathbf{X}}$ and $\mathbf{R}_{\mathbf{YY}} = \mathbf{A}\mathbf{R}_{\mathbf{XX}}\mathbf{A}^T$. Gaussians are among the very few distributions that keep their type under linear maps. (Taking $\mathbf{A}$ = a row selecting one component shows that marginals are Gaussian.)

### 15.1 Lecture exercise — the bivariate pdf from the general formula

Take $n = 2$, $\mathbf{x}-\boldsymbol{\mu} = [u, v]^T$ with $u = x - \mu_X$, $v = y - \mu_Y$, and the $2\times2$ covariance matrix of §14.

1. **Determinant:** $\det\mathbf{R} = \sigma_X^2\sigma_Y^2 - \rho^2\sigma_X^2\sigma_Y^2 = \sigma_X^2\sigma_Y^2(1-\rho^2)$.
2. **Prefactor:** $(2\pi)^{-1}(\det\mathbf{R})^{-1/2} = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}$. ✓
3. **Inverse** ($2\times2$ rule: swap diagonal, negate off-diagonal, divide by det):
$$\mathbf{R}^{-1} = \frac{1}{\sigma_X^2\sigma_Y^2(1-\rho^2)}\begin{bmatrix}\sigma_Y^2 & -\rho\sigma_X\sigma_Y\\ -\rho\sigma_X\sigma_Y & \sigma_X^2\end{bmatrix}.$$
4. **Quadratic form:**
$$[u\ v]\,\mathbf{R}^{-1}\begin{bmatrix}u\\v\end{bmatrix} = \frac{\sigma_Y^2u^2 - 2\rho\sigma_X\sigma_Yuv + \sigma_X^2v^2}{\sigma_X^2\sigma_Y^2(1-\rho^2)} = \frac{1}{1-\rho^2}\Big[\frac{u^2}{\sigma_X^2} - 2\rho\frac{uv}{\sigma_X\sigma_Y} + \frac{v^2}{\sigma_Y^2}\Big].$$
5. Multiplying by $-\frac12$ gives exactly the exponent $-q(x,y)/(2(1-\rho^2))$ of §13. ✓

---

## 16. Problem-solving recipes for Tutorial 2

**Recipe A — two discrete RVs from a counting experiment (Problem 1):**

1. List all equally likely elementary outcomes (here the $\binom{5}{2} = 10$ unordered pairs of cards).
2. Compute $(x, y)$ for each outcome; accumulate the **joint table** (probability $1/10$ per outcome).
3. Row sums → $P_X$, column sums → $P_Y$. Check both sum to 1.
4. Independence: look for a zero cell with positive marginals.
5. Conditional $P_{Y\mid X}$: divide each row by its row sum.
6. Moments: $\mu$, $E\lbrace\cdot^2\rbrace$, variance by shortcut; $E\lbrace XY\rbrace$ over the outcomes; $\sigma_{XY} = \mu_{XY} - \mu_X\mu_Y$; $\rho = \sigma_{XY}/(\sigma_X\sigma_Y)$.
7. $E\lbrace Y\mid X = x\rbrace$ from each conditional row; verify $E\lbrace E\lbrace Y\mid X\rbrace\rbrace = \mu_Y$.

**Recipe B — hierarchical model (Problem 2):**

1. Write $f_X$ and $f_{Y\mid X}$ **with their supports** (indicator conditions).
2. Joint $= f_{Y\mid X}\,f_X$ on the intersection of the conditions. Sketch the support.
3. Marginal of $Y$: for fixed $y$, find the $x$-range where the joint is non-zero (horizontal line!). Split into cases when that range changes form.
4. Check normalisation.

**Recipe C — uniform pdf on a region (Problem 3):**

1. $a = 1/\text{area}$.
2. Boundary equations; marginals = $a\times$ slice length (with ranges, splitting at corners).
3. Conditionals = uniform on the slice: $1/\text{(slice length)}$.
4. Moments by integrals, centroid formulas, or symmetry. $\rho = \sigma_{XY}/(\sigma_X\sigma_Y)$.
5. Interpret: $\rho = 0$ with a non-rectangular support ⇒ uncorrelated but dependent.

---

## 17. Cheat sheet

| Topic | Discrete | Continuous |
|---|---|---|
| joint | $P_{XY}(x,y)$, $\sum\sum = 1$ | $f_{XY}(x,y)$, $\iint = 1$ |
| marginal | $P_X(x) = \sum_yP_{XY}(x,y)$ | $f_X(x) = \int f_{XY}(x,y)\,dy$ |
| conditional | $P_{Y\mid X} = P_{XY}/P_X$ | $f_{Y\mid X} = f_{XY}/f_X$ |
| chain rule | $P_{XY} = P_{Y\mid X}P_X$ | $f_{XY} = f_{Y\mid X}f_X$ |
| independence | $P_{XY} = P_XP_Y$ | $f_{XY} = f_Xf_Y$ (rectangular support) |
| expectation | $\sum\sum g\,P_{XY}$ | $\iint g\,f_{XY}$ |

| Quantity | Formula |
|---|---|
| cdf / pdf | $F_X(x) = P(X\le x) = \int_{-\infty}^xf_X$; $f_X = F_X'$; $P(a<X\le b) = F_X(b) - F_X(a)$ |
| correlation | $\mu_{XY} = E\lbrace XY\rbrace$ |
| covariance | $\sigma_{XY} = \mu_{XY} - \mu_X\mu_Y$; $\mathrm{var}(X+Y) = \sigma_X^2 + \sigma_Y^2 + 2\sigma_{XY}$ |
| correlation coefficient | $\rho_{XY} = \sigma_{XY}/(\sigma_X\sigma_Y)\in[-1,1]$ |
| LMMSE | $\hat Y = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(X - \mu_X)$, MSE $= \sigma_Y^2(1-\rho^2)$ |
| tower property | $E\lbrace Y\rbrace = E\lbrace E\lbrace Y\mid X\rbrace\rbrace$; $E\lbrace XY\rbrace = E\lbrace X\,E\lbrace Y\mid X\rbrace\rbrace$ |
| $E\lbrace Y\mid X\rbrace$ constant | ⇒ uncorrelated |
| independent ⇒ uncorrelated | not conversely (except jointly Gaussian) |
| uniform $U[a,b]$ | $\frac{1}{b-a}$, mean $\frac{a+b}{2}$, var $\frac{(b-a)^2}{12}$ |
| Gaussian | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$; $F = \Phi(\frac{x-\mu}{\sigma})$; $Q = 1-\Phi = \frac12\mathrm{erfc}(\cdot/\sqrt2)$ |
| CLT | $\frac{1}{\sqrt N}\sum(X_n-\mu)\to\mathcal{N}(0,\sigma^2)$ |
| covariance matrix | $\mathbf{R} = E\lbrace\mathbf{XX}^T\rbrace - \boldsymbol{\mu\mu}^T$, symmetric, PSD |
| linear map | $\mathbf{Y} = \mathbf{AX}$: $\boldsymbol{\mu}_Y = \mathbf{A}\boldsymbol{\mu}_X$, $\mathbf{R}_{YY} = \mathbf{AR}_{XX}\mathbf{A}^T$ |
| multivariate Gaussian | $\frac{1}{(2\pi)^{n/2}\sqrt{\det\mathbf{R}}}\exp\lbrace-\frac12(\mathbf{x}-\boldsymbol\mu)^T\mathbf{R}^{-1}(\mathbf{x}-\boldsymbol\mu)\rbrace$ |
