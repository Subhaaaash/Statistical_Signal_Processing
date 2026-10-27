"""
Tutorial 1 -- numerical and symbolic checks of every result in 02_Solutions.md.

Problem 1  Monty Hall: exact Bayes computation + Monte Carlo simulation
Problem 2  Poisson(4) probabilities
Problem 3  Mean and variance of a Poisson RV (symbolic sums)
Problem 4  Binomial pmf = distribution of a sum of n independent Bernoullis
Problem 5  Channel decoder: retransmission probability, geometric law, mean

Run:  python tutorial1_verify.py      (needs numpy scipy sympy matplotlib)
"""
from fractions import Fraction
from itertools import product
from math import comb, exp, factorial
from pathlib import Path

import numpy as np
import sympy as sp

rng = np.random.default_rng(2026)
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def header(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------------------------------------------------------------------
header("Problem 1 - Monty Hall")
# Exact Bayes: contestant picks door 3, host opens door 1.
prior = {d: Fraction(1, 3) for d in (1, 2, 3)}
# P(host opens door 1 | car behind door d), contestant holds door 3
lik = {1: Fraction(0), 2: Fraction(1), 3: Fraction(1, 2)}
p_h1 = sum(lik[d] * prior[d] for d in prior)
post = {d: lik[d] * prior[d] / p_h1 for d in prior}
print("P(host opens 1)                 =", p_h1)
print("posterior P(car = d | host opens 1):", {d: str(v) for d, v in post.items()})
assert post[2] == Fraction(2, 3) and post[3] == Fraction(1, 3)

# Monte Carlo
N = 200_000
car = rng.integers(1, 4, N)
pick = rng.integers(1, 4, N)
host = np.empty(N, dtype=int)
for i in range(N):
    options = [d for d in (1, 2, 3) if d != pick[i] and d != car[i]]
    host[i] = options[rng.integers(len(options))]
switch = 6 - pick - host  # the remaining closed door
print(f"simulation: P(win | stay)   = {np.mean(pick == car):.4f}  (exact 1/3)")
print(f"simulation: P(win | switch) = {np.mean(switch == car):.4f}  (exact 2/3)")

# ---------------------------------------------------------------------------
header("Problem 2 - Poisson(4) website hits")
lam = 4


def pois(k, lam=lam):
    return lam**k * exp(-lam) / factorial(k)


p_ge1 = 1 - pois(0)
p_ge3 = 1 - sum(pois(k) for k in range(3))
p_le5 = sum(pois(k) for k in range(6))
print(f"P(X>=1) = 1 - e^-4          = {p_ge1:.5f}")
print(f"P(X>=3) = 1 - 13 e^-4       = {p_ge3:.5f}")
print(f"P(X<=5) = (643/15) e^-4     = {p_le5:.5f}")
assert abs(p_ge3 - (1 - 13 * exp(-4))) < 1e-12
assert abs(p_le5 - Fraction(643, 15) * exp(-4)) < 1e-12
sim = rng.poisson(lam, 1_000_000)
print(f"simulation:  {np.mean(sim >= 1):.5f}  {np.mean(sim >= 3):.5f}  {np.mean(sim <= 5):.5f}")

# ---------------------------------------------------------------------------
header("Problem 3 - mean and variance of Poisson(lambda)")
k = sp.symbols("k", integer=True, nonnegative=True)
L = sp.symbols("lambda", positive=True)
pmf = L**k * sp.exp(-L) / sp.factorial(k)
total = sp.simplify(sp.summation(pmf, (k, 0, sp.oo)))
mean = sp.simplify(sp.summation(k * pmf, (k, 0, sp.oo)))
fact2 = sp.simplify(sp.summation(k * (k - 1) * pmf, (k, 0, sp.oo)))
var = sp.simplify(fact2 + mean - mean**2)
print("sum of pmf      =", total)
print("E{X}            =", mean)
print("E{X(X-1)}       =", fact2)
print("var(X)          =", var)
assert total == 1 and sp.simplify(mean - L) == 0 and sp.simplify(var - L) == 0

# ---------------------------------------------------------------------------
header("Problem 4 - Binomial as a sum of Bernoullis")
n, p = 5, Fraction(3, 10)
dist = {x: Fraction(0) for x in range(n + 1)}
for seq in product((0, 1), repeat=n):  # enumerate all 2^n outcome sequences
    prob = Fraction(1)
    for y in seq:
        prob *= p if y else 1 - p
    dist[sum(seq)] += prob
for x in range(n + 1):
    formula = comb(n, x) * p**x * (1 - p) ** (n - x)
    print(f"x={x}: enumeration {str(dist[x]):>12}   formula {str(formula):>12}")
    assert dist[x] == formula

# ---------------------------------------------------------------------------
header("Problem 5 - channel decoder with 8-bit codewords")
ps = sp.symbols("p", positive=True)
q = sp.binomial(8, 3) * ps**3 * (1 - ps) ** 5
print("a) P(retransmission) q =", sp.factor(q))
nR = sp.symbols("n_R", integer=True, nonnegative=True)
pmf_nR = q**nR * (1 - q)
print("b) P(N_R = n_R) = q^n_R (1-q)")
Q = sp.symbols("q", positive=True)
EN = sp.piecewise_fold(sp.summation(nR * Q**nR * (1 - Q), (nR, 0, sp.oo)))
EN = EN.args[0][0] if isinstance(EN, sp.Piecewise) else EN  # branch valid for 0 < q < 1
print("c) E{N_R} =", sp.simplify(EN), "  (valid for 0 < q < 1, q from a)")
assert sp.simplify(EN - Q / (1 - Q)) == 0

for pv in (1e-3, 1e-2, 0.1):
    qv = float(q.subs(ps, pv))
    p_corr = sum(comb(8, j) * pv**j * (1 - pv) ** (8 - j) for j in range(3))
    p_fail = 1 - p_corr - qv
    # simulate number of retransmissions
    M = 200_000
    errs = rng.binomial(8, pv, size=(M, 60))
    first_non_retx = np.argmax(errs != 3, axis=1)  # index of first transmission that is not a retx
    print(
        f"p={pv:<6} q={qv:.4e}  E{{N_R}}=q/(1-q)={qv/(1-qv):.4e}  sim={first_non_retx.mean():.4e}"
        f"  | P(correct)={p_corr:.6f}  P(undetected)={p_fail:.3e}"
    )

# ---------------------------------------------------------------------------
header("Figures")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.arange(0, 15)
pm = np.array([pois(int(v)) for v in x])
fig, ax = plt.subplots(figsize=(7, 3.4))
colors = ["#c44e52" if v == 0 else ("#4c72b0" if v <= 5 else "#9aa5b1") for v in x]
ax.bar(x, pm, color=colors)
ax.set_xlabel("number of hits $x$ in 5 minutes")
ax.set_ylabel("$P_X(x)$")
ax.set_title(r"Poisson($\lambda=4$): red = no hit, blue = at most 5 hits ($P(X\leq5)\approx0.785$)")
ax.set_xticks(x)
fig.tight_layout()
fig.savefig(FIG / "poisson_lambda4.png", dpi=150)

pv = np.logspace(-4, -0.5, 200)
qv = 56 * pv**3 * (1 - pv) ** 5
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.loglog(pv, qv, label=r"$q = P(\mathrm{retransmission}) = 56p^3(1-p)^5$")
ax.loglog(pv, qv / (1 - qv), "--", label=r"$E\{N_R\} = q/(1-q)$")
ax.set_xlabel("bit error probability $p$")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_title("Problem 5: retransmissions vs bit error probability")
fig.tight_layout()
fig.savefig(FIG / "retransmissions.png", dpi=150)
print("saved figures to", FIG)
