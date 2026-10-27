---
title: "Week 1 — Concepts"
subtitle: "Statistical Signal Processing · Lecture 1: Introduction + Probability Basics for Tutorial 1"
---

# Week 1 — Concepts

**Course:** Statistical Signal Processing (L.048.24014, Universität Paderborn, Signal and System Theory Group, Dr.-Ing. Tanuj Hasija), WiSe 2026/27
**Lecture:** 1 — Introduction, plus the parts of Lecture 2 ("Review of probability theory", §2.1–2.2) that Tutorial 1 relies on
**Covers:** everything needed for Tutorial 1 (Problems 1–5: Monty Hall, Poisson, Binomial, channel decoder)

Lecture 1 is an overview of the course; it contains no formulas. Tutorial 1, however, already needs events, conditional probability, Bayes' rule, discrete random variables, expectation and the Binomial and Poisson distributions. Those come from the first half of the Lecture 2 handout, so they are built up here from scratch. Week 2 continues with the second half (joint, continuous and Gaussian random variables, random vectors).

The worked solutions are in [`02_Solutions.md`](02_Solutions.md). The script [`python/tutorial1_verify.py`](python/tutorial1_verify.py) checks every number (exact fractions, symbolic sums and Monte Carlo simulation).

**Notation** follows the lecture slides:

| Symbol | Meaning |
|---|---|
| $\Omega$ | sample space (set of all outcomes $\omega$) |
| $A, B, E, F$ | events (subsets of $\Omega$) |
| $\overline{A}$ | complement of $A$ ("not $A$") |
| $P(A)$ | probability of the event $A$ |
| $P(AB) = P(A\cap B)$ | joint probability |
| $X, Y$ (capital) | random variables; $x, y$ (small) are the values they take |
| $P_X(x) = P(X = x)$ | probability distribution (pmf) of a discrete RV |
| $E\lbrace\cdot\rbrace$ | expectation operator |
| $\mu_X$, $\sigma_X^2$ | mean and variance of $X$ |
| $\binom{n}{k}$ | binomial coefficient "$n$ choose $k$" |

---

## Contents

1. [Lecture 1: what statistical signal processing is about](#1-lecture-1-what-statistical-signal-processing-is-about)
2. [Sample spaces, events and set operations](#2-sample-spaces-events-and-set-operations)
3. [The axioms of probability and what follows from them](#3-the-axioms-of-probability-and-what-follows-from-them)
4. [Equally likely outcomes and counting](#4-equally-likely-outcomes-and-counting)
5. [Conditional probability and independence](#5-conditional-probability-and-independence)
6. [Total probability and Bayes' rule](#6-total-probability-and-bayes-rule)
7. [Discrete random variables](#7-discrete-random-variables)
8. [Expectation, variance and moments](#8-expectation-variance-and-moments)
9. [Several random variables and sums of independent RVs](#9-several-random-variables-and-sums-of-independent-rvs)
10. [Important discrete distributions](#10-important-discrete-distributions)
11. [Series toolbox](#11-series-toolbox)
12. [Problem-solving recipes for Tutorial 1](#12-problem-solving-recipes-for-tutorial-1)
13. [Cheat sheet](#13-cheat-sheet)

---

## 1. Lecture 1: what statistical signal processing is about

### 1.1 Definition

**Statistical signal processing (SSP)** deals with signals that are described by **statistical models**. Real measurements are imperfect: they contain noise, they are incomplete, and the system that produced them is only partly known. SSP describes this uncertainty with probability and uses the description to **draw inferences** (make decisions, estimate numbers, predict) from the data. (Definition after Scharf, 1991.)

The word "statistical" is the key: instead of asking "what is the signal?" we ask "what is the **most probable** explanation of what we measured, and how sure can we be?". This is why the course starts with a review of probability theory.

### 1.2 The three branches

| Branch | Question it answers | Examples from the lecture |
|---|---|---|
| **Detection** (hypothesis testing) | In which of a **finite number of states** is the system? | radar/sonar target present or not; decoding bits in digital communication; speech recognition; image classification; **seizure monitoring**: no seizure / pre-seizure / seizure / post-seizure |
| **Estimation** (parameter estimation) | What is the **value** of an unknown quantity? | range and Doppler shift in radar; source directions with sensor arrays; channel characterisation; **autonomous driving** (positions and velocities of other cars); **brain activity from fMRI** |
| **Time series analysis** | How does a signal **behave over time**? Predict, filter, smooth, analyse its spectrum | tracking aircraft and vehicles; resonant modes in geophysical data; **EEG** analysis |

The branches overlap. The seizure-monitoring example in the slides combines all of them: wearable sensors record time series (EDA, heart rate, temperature), signal quality is scored, features are estimated (with an LSTM and a deep autoencoder), and finally a detector decides "pre-seizure or not".

**Difference between detection and estimation:** detection picks one answer from a finite list (a decision), estimation produces a number from a continuum. Tutorial 1, Problem 5 is a tiny detection example: the channel decoder decides between "correct", "retransmit" and (unknowingly) "wrong".

### 1.3 Course outline

1. Introduction
2. Review of probability theory ← Week 1 and Week 2 notes
3. Review of linear algebra (matrix analysis)
4. Complex-valued random variables
5. Estimation: maximum likelihood; minimum variance distortionless response (MVDR); MMSE and linear MMSE; reduced-rank estimation
6. Detection: Neyman–Pearson detectors; simple Gaussian hypothesis testing
7. Wide-sense stationary random processes: LTI systems with random inputs; Wiener filter

### 1.4 Literature

- Gubner, *Probability and Random Processes for Electrical and Computer Engineers* (2006) — refresher on basic probability (the best match for Weeks 1–2).
- Scharf, *Statistical Signal Processing: Detection, Estimation, and Time Series Analysis* (1991) — broad, mathematically demanding.
- Schreier & Scharf, *Statistical Signal Processing of Complex-Valued Data* (2010) — complex data.
- Kay, *Fundamentals of Statistical Signal Processing*, Vol. I (Estimation) and Vol. II (Detection) (1993) — very readable.
- Gray & Davisson, *An Introduction to Statistical Signal Processing* (2010) — basic concepts.

### 1.5 Organisation

- Lecture Wed 9:15–10:45, tutorial Wed 11:00–12:30, room P1.6.11. Material on PANDA.
- **Exam:** written, 2 hours, closed book, but **one double-sided handwritten A4 formula sheet** is allowed. The cheat sheets at the end of these notes are written with that sheet in mind.

---

## 2. Sample spaces, events and set operations

A **random experiment** is a procedure whose result is not known in advance (rolling a die, transmitting a bit over a noisy channel).

- The **sample space** $\Omega$ is the set of **all possible distinct outcomes** $\omega$. Die: $\Omega = \lbrace 1,2,3,4,5,6\rbrace$.
- An **event** is a set (collection) of outcomes, i.e. a subset of $\Omega$. "Roll an even number": $E = \lbrace 2,4,6\rbrace$. We say the event **occurs** if the outcome $\omega$ lies in it.

Set operations and their meaning in words:

| Operation | Definition | In words |
|---|---|---|
| complement $\overline{E}$ | $\lbrace\omega\in\Omega : \omega\notin E\rbrace$ | "$E$ does **not** occur" |
| empty set $\emptyset = \overline{\Omega}$ | no outcomes | the impossible event |
| union $E\cup F$ | outcomes in $E$ **or** $F$ (or both) | "at least one of them occurs" |
| intersection $E\cap F$ (also $EF$) | outcomes in $E$ **and** $F$ | "both occur" |

$E$ and $F$ are **mutually exclusive (disjoint)** if $E\cap F = \emptyset$: they cannot occur together. Several events $E_1, E_2, \dots$ are **pairwise disjoint** if $E_i\cap E_j = \emptyset$ for all $i\ne j$.

Two facts used constantly (Venn diagrams make them obvious):

- **Splitting an event:** $E = (E\cap F)\cup(E\cap\overline{F})$, and the two pieces are disjoint.
- **De Morgan's laws:** $\overline{E\cup F} = \overline{E}\cap\overline{F}$ and $\overline{E\cap F} = \overline{E}\cup\overline{F}$. In words: "not (at least one)" = "none".

A collection $E_1,\dots,E_K$ is a **partition** of $\Omega$ if the events are pairwise disjoint and $E_1\cup\dots\cup E_K = \Omega$: exactly one of them occurs. Example: "the car is behind door 1 / 2 / 3" (Tutorial 1, Problem 1).

---

## 3. The axioms of probability and what follows from them

### 3.1 The three axioms

Probability is a **set function** $P(\cdot)$ that assigns to every event $E$ a number $P(E)$ such that

1. $P(E)\ge0$ (non-negativity),
2. $P(\Omega) = 1$ (something happens),
3. $P(E\cup F) = P(E) + P(F)$ **if** $E\cap F = \emptyset$ (additivity for disjoint events). This extends to any finite or countable list of pairwise disjoint events.

This is an idealisation: the axioms do not say what probability "is", only the rules it obeys. Everything else is derived from them.

### 3.2 Consequences (each one line from the axioms)

- **Complement rule:** $\Omega = E\cup\overline{E}$ (disjoint) ⇒ $1 = P(E) + P(\overline{E})$ ⇒
$$P(\overline{E}) = 1 - P(E).$$
This is the most useful trick in Tutorial 1: "**at least one**" is the complement of "**none**", and "none" is usually easy.
- $P(\emptyset) = 1 - P(\Omega) = 0$.
- If $E\subseteq F$ then $P(E)\le P(F)$, and always $0\le P(E)\le1$.

### 3.3 Union of events that are not disjoint (derivation from the lecture)

Write $E\cup F$ as three **disjoint** pieces: $E\cup F = (E\cap\overline{F})\cup(E\cap F)\cup(\overline{E}\cap F)$. Axiom 3 gives
$$P(E\cup F) = P(E\overline{F}) + P(EF) + P(\overline{E}F).$$
Splitting $E$ and $F$: $P(E\overline{F}) + P(EF) = P(E)$ and $P(\overline{E}F) + P(EF) = P(F)$. Substitute $P(E\overline{F}) = P(E) - P(EF)$ and $P(\overline{E}F) = P(F) - P(EF)$:
$$\boxed{P(E\cup F) = P(E) + P(F) - P(E\cap F)}$$
The intersection is subtracted once because it was counted twice. Consequence (**union bound**): $P(E_1\cup\dots\cup E_K)\le\sum_i P(E_i)$, with equality for pairwise disjoint events.

---

## 4. Equally likely outcomes and counting

### 4.1 Classical probability

If $\Omega$ is finite and all outcomes are **equally likely**, then
$$P(E) = \frac{\text{number of outcomes in }E}{\text{number of outcomes in }\Omega} = \frac{|E|}{|\Omega|}.$$
So probability becomes counting. The danger is choosing a sample space whose outcomes are **not** equally likely (classic mistake: "the sum of two dice is 2, 3, …, 12, so each has probability 1/11" — wrong, see the lecture table where $P_Y(7) = 6/36$).

### 4.2 Counting rules

- **Multiplication principle:** if a task consists of $k$ steps with $n_1, n_2, \dots, n_k$ possibilities each, there are $n_1n_2\cdots n_k$ ways. Three coin tosses: $2^3 = 8$ outcomes. An 8-bit word: $2^8 = 256$ error patterns.
- **Ordered selections without repetition:** choosing $k$ of $n$ objects in order: $n(n-1)\cdots(n-k+1) = \frac{n!}{(n-k)!}$.
- **Unordered selections (combinations):** choosing a $k$-element **subset** of $n$ objects:
$$\binom{n}{k} = \frac{n!}{k!\,(n-k)!}.$$
Reason: each subset can be ordered in $k!$ ways, so (ordered count)/$k!$.

Useful values and properties:

- $\binom{n}{0} = \binom{n}{n} = 1$, $\binom{n}{1} = n$, $\binom{n}{k} = \binom{n}{n-k}$.
- $\binom{8}{0},\dots,\binom{8}{4} = 1, 8, 28, 56, 70$ (needed in Problem 5); $\binom{5}{2} = 10$ (Tutorial 2, Problem 1).
- **Pascal's rule:** $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$ (either object $n$ is in the subset or not).
- **Binomial theorem:** $(a+b)^n = \sum_{k=0}^{n}\binom{n}{k}a^kb^{n-k}$. With $a = p$, $b = 1-p$ it shows that the Binomial pmf sums to 1.

**Why $\binom{n}{x}$ counts bit patterns:** a sequence of $n$ trials with exactly $x$ "successes" is determined by **which** $x$ positions are successes, i.e. by an $x$-element subset of $\lbrace1,\dots,n\rbrace$. There are $\binom{n}{x}$ such subsets. This is the heart of Problem 4.

---

## 5. Conditional probability and independence

### 5.1 Conditional probability

The probability of $E$ **given** that $F$ has occurred ($P(F)>0$):
$$P(E\mid F) \triangleq \frac{P(EF)}{P(F)}.$$
Interpretation: knowing $F$ happened shrinks the sample space to $F$; we renormalise so that $F$ has probability 1, and keep only the part of $E$ that lies in $F$.

Rearranged, it is the **multiplication (chain) rule**:
$$P(EF) = P(E\mid F)\,P(F) = P(F\mid E)\,P(E),$$
and for more events $P(E_1E_2E_3) = P(E_1)\,P(E_2\mid E_1)\,P(E_3\mid E_1E_2)$. This is how sequential experiments (draw a coin, then look at it; pick a door, then the host opens one) are modelled: multiply along the branches of a **probability tree**.

$P(\cdot\mid F)$ is itself a probability (it satisfies the three axioms), so all rules carry over, e.g. $P(\overline{E}\mid F) = 1 - P(E\mid F)$.

### 5.2 Statistical independence of events

$E$ and $F$ are **statistically independent** if
$$P(EF) = P(E)\,P(F).$$
Then $P(E\mid F) = \frac{P(E)P(F)}{P(F)} = P(E)$ and $P(F\mid E) = P(F)$: knowing one tells you nothing about the other.

Notes:

- Independent is **not** the same as disjoint. Disjoint events with positive probabilities are strongly **dependent**: if one occurs, the other cannot ($P(E\mid F) = 0\ne P(E)$).
- Several events $E_1,\dots,E_n$ are (mutually) independent if the product rule holds for **every** sub-collection: $P(E_{i_1}\cdots E_{i_k}) = P(E_{i_1})\cdots P(E_{i_k})$.
- **Independent trials:** "bit errors occur independently with probability $p$" means the probability of any specific error pattern is the product of the per-bit probabilities. For example the pattern "error, ok, error" has probability $p(1-p)p$. This is used in Problems 4 and 5.
- If $E$ and $F$ are independent, so are $E$ and $\overline{F}$, $\overline{E}$ and $F$, $\overline{E}$ and $\overline{F}$.

### 5.3 Lecture example 1 — three fair coin tosses

$\Omega = \lbrace TTT, TTH, THT, THH, HTT, HTH, HHT, HHH\rbrace$, all 8 equally likely.

- $A$ = "at least two heads" $= \lbrace THH, HTH, HHT, HHH\rbrace$ ⇒ $P(A) = 4/8 = 1/2$.
- $B$ = "heads and tails alternate" $= \lbrace HTH, THT\rbrace$ ⇒ $P(B) = 2/8 = 1/4$.
- $AB = \lbrace HTH\rbrace$ ⇒ $P(AB) = 1/8$.

Since $P(A)P(B) = \frac12\cdot\frac14 = \frac18 = P(AB)$, $A$ and $B$ **are independent** — even though they are defined from the same tosses. Independence is a numerical property, not something you can always "see".

---

## 6. Total probability and Bayes' rule

### 6.1 Law of total probability

Let $E_1,\dots,E_K$ be a **partition** of $\Omega$ (pairwise disjoint, together everything). For any event $F$, the pieces $F\cap E_i$ are disjoint and make up $F$, so
$$\boxed{P(F) = \sum_{i=1}^{K}P(F\mid E_i)\,P(E_i)}$$
"Average the conditional probabilities over all the cases, weighted by how likely each case is." The two-case version uses the partition $\lbrace E,\overline{E}\rbrace$: $P(F) = P(F\mid E)P(E) + P(F\mid\overline{E})P(\overline{E})$.

### 6.2 Bayes' rule

Combine the two forms of the multiplication rule, $P(E_j\mid F)P(F) = P(F\mid E_j)P(E_j)$, and expand $P(F)$ with total probability:
$$\boxed{P(E_j\mid F) = \frac{P(F\mid E_j)\,P(E_j)}{\sum_{i=1}^{K}P(F\mid E_i)\,P(E_i)}}$$
Vocabulary (used throughout detection and estimation later in the course):

| Term | Symbol | Meaning |
|---|---|---|
| prior | $P(E_j)$ | belief about case $E_j$ **before** the observation |
| likelihood | $P(F\mid E_j)$ | how probable the observation $F$ is **if** case $E_j$ is true |
| evidence | $P(F)$ | total probability of the observation (normaliser) |
| posterior | $P(E_j\mid F)$ | updated belief **after** observing $F$ |

**Bayes' rule reverses the conditioning:** we usually know $P(\text{observation}\mid\text{cause})$ from a model, and want $P(\text{cause}\mid\text{observation})$. That is exactly the situation in Monty Hall (we know how the host behaves given the car's position; we want the car's position given what the host did) and in every detection problem.

A shortcut: since the denominator is the same for all $j$, $P(E_j\mid F)\propto P(F\mid E_j)P(E_j)$. Compute the numerators, then divide by their sum.

### 6.3 Lecture example 2 — the fake coin

Five coins, one has heads on both sides. Draw one at random, put it on the table. Heads shows. What is the probability that the other side is also heads?

Use the 10 equally likely (coin, side facing up) outcomes:

| coin | 1 | 2 | 3 | 4 | 5 (fake) |
|---|---|---|---|---|---|
| side 1 | H | H | H | H | H |
| side 2 | T | T | T | T | H |

- $A$ = "head shows": 4 normal coins with side 1 up + fake coin either side up = 6 outcomes ⇒ $P(A) = 6/10 = 3/5$.
- $B$ = "head on the other side": normal coins with T up (4 outcomes) + fake coin (2 outcomes) ⇒ $P(B) = 6/10 = 3/5$.
- $AB$ = head shows **and** head below: only the fake coin, 2 outcomes ⇒ $P(AB) = 2/10 = 1/5$.
- $P(B\mid A) = \frac{P(AB)}{P(A)} = \frac{1/5}{3/5} = \boxed{1/3}$.

Intuition check: among the 6 "head-up" faces, 2 belong to the fake coin. The naive answer "1/5, since one coin of five is fake" ignores that **observing heads is evidence for the fake coin** (it shows heads twice as often). Same lesson as Monty Hall: the observation carries information, and the way it was generated matters.

---

## 7. Discrete random variables

### 7.1 Definition

A **random variable (RV)** $X(\omega)$ is a **function** that assigns a real number to every outcome $\omega\in\Omega$. It is neither random nor a variable in the everyday sense: the randomness is in $\omega$; $X$ just converts outcomes into numbers so that we can compute with them. We usually drop $\omega$ and write $X$; the capital letter is the RV, the small letter $x$ a value it can take.

**Lecture example:** $X$ = number of heads in three tosses. $X(TTT) = 0$, $X = 1$ on $\lbrace TTH, THT, HTT\rbrace$, $X = 2$ on $\lbrace THH, HTH, HHT\rbrace$, $X(HHH) = 3$. Each value inherits the probability of the outcomes mapped to it:

| $x$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $P_X(x)$ | 1/8 | 3/8 | 3/8 | 1/8 |

and e.g. $P(X\ge2) = 3/8 + 1/8 = 1/2$.

### 7.2 Discrete RVs and the probability distribution (pmf)

A RV is **discrete** if it takes only a **countable** number of values $x_1, x_2, \dots$ (finite, or infinite like $0,1,2,\dots$). Examples: a die; the number of cars crossing a bridge in one hour; the number of website hits in 5 minutes. (Continuous RVs — an angle, a waiting time — come in Week 2.)

The **probability distribution** (probability mass function, pmf) is
$$P_X(x)\triangleq P(X = x),\qquad 0\le P_X(x)\le1,\qquad \sum_x P_X(x) = 1.$$
Probabilities of sets of values are sums: $P(X\in S) = \sum_{x\in S}P_X(x)$. For example, for integer-valued $X$: $P(a\le X\le b) = P(X\le b) - P(X\le a-1)$.

**Lecture example (two dice, $Y$ = sum):** $P_Y(y) = \frac{6 - |y-7|}{36}$ for $y = 2,\dots,12$, so $P(Y\le4) = \frac{1+2+3}{36} = \frac16$ and $P(4\le Y\le6) = P(Y\le6) - P(Y\le3) = \frac{15}{36} - \frac{3}{36} = \frac13$.

The **cumulative distribution function** $F_X(x) = P(X\le x)$ of a discrete RV is a staircase that jumps by $P_X(x_i)$ at each value $x_i$.

---

## 8. Expectation, variance and moments

### 8.1 Mean (expected value)

$$E\lbrace X\rbrace = \mu_X\triangleq\sum_x x\,P_X(x).$$
It is the probability-weighted average of the values, i.e. the "centre of mass" of the pmf, and the long-run average of many independent repetitions. **$E\lbrace X\rbrace$ is a number, not a random variable.** For an infinite sum, the mean exists only if the series converges absolutely.

**Expectation of a function** (no need to find the pmf of $g(X)$ first):
$$E\lbrace g(X)\rbrace = \sum_x g(x)\,P_X(x).$$

### 8.2 Linearity

For constants $a, b$ and any RVs $X, Y$ (independent or not):
$$E\lbrace aX + bY\rbrace = aE\lbrace X\rbrace + bE\lbrace Y\rbrace,\qquad E\lbrace c\rbrace = c.$$
Consequence: the mean of a sum is the sum of the means. For $X = Y_1+\dots+Y_n$ this gives $E\lbrace X\rbrace = \sum E\lbrace Y_k\rbrace$ immediately (Binomial mean, §10.2).

### 8.3 Variance and standard deviation

$$\mathrm{var}(X) = \sigma_X^2\triangleq E\lbrace(X-\mu_X)^2\rbrace = \sum_x(x-\mu_X)^2P_X(x),\qquad \sigma_X = \sqrt{\mathrm{var}(X)}.$$
The variance measures the spread around the mean (in squared units; $\sigma_X$ is in the units of $X$).

**Shortcut formula** — expand the square and use linearity ($\mu_X$ is a constant):
$$E\lbrace(X-\mu_X)^2\rbrace = E\lbrace X^2\rbrace - 2\mu_XE\lbrace X\rbrace + \mu_X^2 = E\lbrace X^2\rbrace - \mu_X^2.$$
$$\boxed{\sigma_X^2 = \mu_{X^2} - \mu_X^2},\qquad \mu_{X^2} = E\lbrace X^2\rbrace\ \text{(second-order moment)}.$$

Rules: $\mathrm{var}(aX + b) = a^2\,\mathrm{var}(X)$ (shifting does not change spread; scaling by $a$ scales $\sigma$ by $|a|$), and $\mathrm{var}(X)\ge0$ with equality only if $X$ is constant.

### 8.4 The factorial-moment trick

For RVs on $0,1,2,\dots$ whose pmf contains $x!$ (Poisson, Binomial), $E\lbrace X^2\rbrace$ is awkward but $E\lbrace X(X-1)\rbrace$ is easy, because $x(x-1)/x! = 1/(x-2)!$ cancels nicely. Then
$$E\lbrace X^2\rbrace = E\lbrace X(X-1)\rbrace + E\lbrace X\rbrace,\qquad \sigma_X^2 = E\lbrace X(X-1)\rbrace + \mu_X - \mu_X^2.$$
This is the cleanest route to the Poisson variance (Problem 3).

### 8.5 Higher-order moments and why moments matter

- Skewness $E\lbrace((X-\mu_X)/\sigma_X)^3\rbrace$ measures asymmetry; kurtosis $E\lbrace X^4\rbrace/\sigma_X^4 - 3$ (for zero-mean $X$) measures how concentrated/heavy-tailed a distribution is compared with the Gaussian.
- Moments **summarise** a distribution with a few numbers. Only the full distribution is a complete description, but in practice moments are often all we know or can estimate.

---

## 9. Several random variables and sums of independent RVs

Week 2 treats joint distributions in detail; Tutorial 1 needs only the following.

- Discrete RVs $Y_1,\dots,Y_n$ are **independent** if their joint pmf factorises: $P(Y_1 = y_1,\dots,Y_n = y_n) = P_{Y_1}(y_1)\cdots P_{Y_n}(y_n)$ for all values. "Independent trials" means exactly this.
- **Mean of a sum** (always): $E\lbrace\sum_kY_k\rbrace = \sum_kE\lbrace Y_k\rbrace$.
- **Variance of a sum** (if independent, more generally if uncorrelated): $\mathrm{var}(\sum_kY_k) = \sum_k\mathrm{var}(Y_k)$. Proof sketch: expand the square; the cross terms $E\lbrace(Y_i-\mu_i)(Y_j-\mu_j)\rbrace = E\lbrace Y_i-\mu_i\rbrace E\lbrace Y_j-\mu_j\rbrace = 0$ for independent $Y_i, Y_j$.
- **Distribution of a sum of two independent integer RVs** (discrete convolution):
$$P(X_1 + X_2 = s) = \sum_k P_{X_1}(k)\,P_{X_2}(s-k).$$
This is the "total probability over the value of $X_1$" argument. It gives a second proof of the Binomial pmf by induction (Problem 4) and shows that the sum of independent Poissons is Poisson ($\lambda_1+\lambda_2$).

---

## 10. Important discrete distributions

### 10.1 Bernoulli($p$)

One trial with two outcomes: $Y = 1$ ("success", e.g. "this bit is in error") with probability $p$, $Y = 0$ with probability $1-p$. Compactly $P_Y(y) = p^y(1-p)^{1-y}$, $y\in\lbrace0,1\rbrace$.

- $E\lbrace Y\rbrace = 0\cdot(1-p) + 1\cdot p = p$.
- $E\lbrace Y^2\rbrace = p$ (since $Y^2 = Y$), so $\mathrm{var}(Y) = p - p^2 = p(1-p)$.

A Bernoulli RV is an **indicator**: it counts whether an event happened. Sums of indicators count how many events happened.

### 10.2 Binomial($n, p$)

Number of successes in $n$ **independent** Bernoulli($p$) trials:
$$P_X(x) = \binom{n}{x}p^x(1-p)^{n-x},\qquad x = 0,1,\dots,n.$$
(The derivation is Tutorial 1, Problem 4.) Properties:

- Sums to 1 by the binomial theorem: $\sum_x\binom{n}{x}p^x(1-p)^{n-x} = (p + 1 - p)^n = 1$.
- $E\lbrace X\rbrace = np$, $\mathrm{var}(X) = np(1-p)$ (sum of $n$ independent Bernoullis, §9).
- "At most $k$": $P(X\le k) = \sum_{x=0}^{k}\binom{n}{x}p^x(1-p)^{n-x}$.

### 10.3 Geometric($q$) — "how many repeats until it stops"

Repeat independent trials; each "fails" (has to be repeated) with probability $q$ and "stops" with probability $1-q$. Let $N$ = number of **failures before the first stop**. The event $\lbrace N = n\rbrace$ is "$n$ failures in a row, then a stop", so by independence
$$P_N(n) = q^n(1-q),\qquad n = 0,1,2,\dots$$
Check: $\sum_{n\ge0}q^n(1-q) = (1-q)\frac{1}{1-q} = 1$. Mean (using §11.2):
$$E\lbrace N\rbrace = \sum_{n\ge0}n\,q^n(1-q) = (1-q)\frac{q}{(1-q)^2} = \frac{q}{1-q}.$$
Also $P(N\ge n) = q^n$ and $\mathrm{var}(N) = q/(1-q)^2$.

Careful with conventions: if instead $M$ counts **all** trials including the final one, $M = N + 1$, $P_M(m) = q^{m-1}(1-q)$ for $m\ge1$, and $E\lbrace M\rbrace = 1/(1-q)$. Always define clearly what is counted. This is Problem 5(b)–(c), where $N$ = number of retransmissions.

### 10.4 Poisson($\lambda$)

$$P_X(x) = \frac{\lambda^xe^{-\lambda}}{x!},\qquad x = 0,1,2,\dots,\quad\lambda>0.$$
It models the **number of events in a fixed interval** when events happen independently at a constant average rate: photons hitting a detector (photoelectric effect), radioactive decays, calls at an exchange, **hits on a website** (Problem 2).

- **Sums to 1:** $\sum_x\frac{\lambda^x}{x!} = e^{\lambda}$ (§11.1), times $e^{-\lambda}$ gives 1.
- **Mean and variance are both $\lambda$** (derived in Problem 3). So $\lambda$ is the **average number of events per interval**; e.g. Poisson(4) for 5 minutes means on average 4 hits per 5 minutes.
- **Rate scaling:** if events arrive at rate $r$ per unit time, the count in an interval of length $T$ is Poisson($rT$). Twice as long an interval ⇒ twice the $\lambda$. Counts in **disjoint** intervals are independent, and the sum of independent Poissons is Poisson with the summed $\lambda$.
- **Where it comes from (Poisson limit of the Binomial):** split the interval into $n$ tiny slots, each containing an event with probability $p = \lambda/n$ independently. The count is Binomial($n,\lambda/n$), and as $n\to\infty$
$$\binom{n}{x}\Big(\frac{\lambda}{n}\Big)^x\Big(1-\frac{\lambda}{n}\Big)^{n-x}\;\to\;\frac{\lambda^x}{x!}e^{-\lambda},$$
because $\frac{n(n-1)\cdots(n-x+1)}{n^x}\to1$, $(1-\lambda/n)^n\to e^{-\lambda}$ and $(1-\lambda/n)^{-x}\to1$. So Poisson = "many trials, each very unlikely". In practice Binomial($n,p$) ≈ Poisson($np$) for large $n$ and small $p$ — useful for bit errors in long codewords.
- **Ratio of successive probabilities:** $\frac{P_X(x)}{P_X(x-1)} = \frac{\lambda}{x}$. This makes hand calculation quick: start from $P_X(0) = e^{-\lambda}$ and multiply by $\lambda/1, \lambda/2, \dots$ It also shows the mode is at $\lfloor\lambda\rfloor$ (and at $\lambda-1$ too when $\lambda$ is an integer: $P_X(3) = P_X(4)$ for $\lambda = 4$).

### 10.5 Summary table

| Distribution | pmf | range | mean | variance |
|---|---|---|---|---|
| Bernoulli($p$) | $p^y(1-p)^{1-y}$ | $\lbrace0,1\rbrace$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{x}p^x(1-p)^{n-x}$ | $0..n$ | $np$ | $np(1-p)$ |
| Geometric($q$), failures before stop | $q^n(1-q)$ | $0,1,\dots$ | $\frac{q}{1-q}$ | $\frac{q}{(1-q)^2}$ |
| Poisson($\lambda$) | $\frac{\lambda^xe^{-\lambda}}{x!}$ | $0,1,\dots$ | $\lambda$ | $\lambda$ |

---

## 11. Series toolbox

### 11.1 Exponential series

$$e^{\lambda} = \sum_{k=0}^{\infty}\frac{\lambda^k}{k!} = 1 + \lambda + \frac{\lambda^2}{2!} + \frac{\lambda^3}{3!} + \cdots$$
**Index-shift trick** (used for Poisson moments): when a sum starts with terms that vanish, drop them and re-index. For example
$$\sum_{x=0}^{\infty}x\frac{\lambda^x}{x!} = \sum_{x=1}^{\infty}\frac{\lambda^x}{(x-1)!} = \lambda\sum_{m=0}^{\infty}\frac{\lambda^m}{m!} = \lambda e^{\lambda}\quad(m = x-1).$$

### 11.2 Geometric series and its derivative

For $|q|<1$:
$$\sum_{n=0}^{\infty}q^n = \frac{1}{1-q},\qquad \sum_{n=0}^{\infty}n\,q^n = \frac{q}{(1-q)^2},\qquad \sum_{n=0}^{\infty}n^2q^n = \frac{q(1+q)}{(1-q)^3}.$$
The second follows by differentiating the first with respect to $q$, $\sum nq^{n-1} = 1/(1-q)^2$, and multiplying by $q$. Finite version: $\sum_{n=0}^{N-1}q^n = \frac{1-q^N}{1-q}$.

### 11.3 Numbers worth knowing

$e^{-4}\approx0.018316$, $e^{-1}\approx0.3679$, $e^{-2}\approx0.1353$. For small $p$: $(1-p)^n\approx1-np$ and $(1-p)^n\approx e^{-np}$.

---

## 12. Problem-solving recipes for Tutorial 1

**Recipe A — Bayes problems (Problem 1, fake coin):**

1. List the possible **causes** (a partition) and their **priors**.
2. Describe the **observation** precisely, including **how it was generated** (who chose what, with which rule). This is where intuition fails.
3. Write the **likelihood** of the observation under each cause.
4. Posterior $\propto$ likelihood × prior; normalise by the sum. Sanity-check that posteriors add to 1.
5. If unsure, enumerate equally likely elementary outcomes in a table or tree, or simulate.

**Recipe B — probabilities from a pmf (Problem 2):**

1. Identify the distribution and its parameter (match the time interval to $\lambda$).
2. Translate the words: "at least $k$" = $P(X\ge k)$; "no more than $k$" = $P(X\le k)$; "more than $k$" = $P(X\ge k+1)$.
3. For "at least", use the **complement**: $P(X\ge k) = 1 - \sum_{x=0}^{k-1}P_X(x)$ (finitely many terms instead of infinitely many).
4. Compute $P_X(0), P_X(1),\dots$ recursively with the ratio $\lambda/x$.

**Recipe C — moments of a named distribution (Problem 3):**

1. Write the definition $E\lbrace X\rbrace = \sum xP_X(x)$.
2. Cancel $x$ against $x!$, shift the index, recognise the series that sums the pmf to 1.
3. For the variance use $E\lbrace X(X-1)\rbrace$, then $\sigma^2 = E\lbrace X(X-1)\rbrace + \mu - \mu^2$.

**Recipe D — deriving a distribution of a sum (Problem 4):**

1. Write the event $\lbrace X = x\rbrace$ as a **union of disjoint elementary events** (all 0/1 sequences with $x$ ones).
2. Probability of one such sequence by **independence** (product).
3. All have the same probability ⇒ multiply by their **number** (counting).
4. Alternative: induction with the convolution formula.

**Recipe E — modelling a protocol (Problem 5):**

1. Model one transmission: what is random (number of bit errors ~ Binomial), which outcomes lead to which action (correct / retransmit / wrong).
2. Repetitions are independent trials ⇒ "number of repeats" is **Geometric**.
3. Mean from the series in §11.2.

---

## 13. Cheat sheet

| Topic | Formula |
|---|---|
| Complement | $P(\overline{A}) = 1 - P(A)$ |
| Union | $P(A\cup B) = P(A) + P(B) - P(AB)$ |
| Conditional | $P(A\mid B) = P(AB)/P(B)$ |
| Chain rule | $P(AB) = P(A\mid B)P(B)$ |
| Independence | $P(AB) = P(A)P(B)$ ⇔ $P(A\mid B) = P(A)$ |
| Total probability | $P(F) = \sum_iP(F\mid E_i)P(E_i)$ ($E_i$ partition) |
| Bayes | $P(E_j\mid F) = \dfrac{P(F\mid E_j)P(E_j)}{\sum_iP(F\mid E_i)P(E_i)}$ |
| Combinations | $\binom{n}{k} = \frac{n!}{k!(n-k)!}$; $\binom{8}{3} = 56$ |
| pmf | $P_X(x)\ge0$, $\sum_xP_X(x) = 1$ |
| Mean | $\mu_X = \sum_xxP_X(x)$; $E\lbrace g(X)\rbrace = \sum_xg(x)P_X(x)$ |
| Linearity | $E\lbrace aX + bY\rbrace = aE\lbrace X\rbrace + bE\lbrace Y\rbrace$ |
| Variance | $\sigma_X^2 = E\lbrace X^2\rbrace - \mu_X^2$; $\mathrm{var}(aX+b) = a^2\sigma_X^2$ |
| Factorial moment | $\sigma_X^2 = E\lbrace X(X-1)\rbrace + \mu_X - \mu_X^2$ |
| Sum of independent | means add; variances add; pmf = convolution |
| Binomial | $\binom{n}{x}p^x(1-p)^{n-x}$, mean $np$, var $np(1-p)$ |
| Geometric | $q^n(1-q)$, mean $q/(1-q)$ |
| Poisson | $\lambda^xe^{-\lambda}/x!$, mean = var = $\lambda$; $P(x)/P(x-1) = \lambda/x$ |
| Series | $e^\lambda = \sum\lambda^k/k!$; $\sum q^n = \frac{1}{1-q}$; $\sum nq^n = \frac{q}{(1-q)^2}$ |
