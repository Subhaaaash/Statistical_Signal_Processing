"""
Tutorial 2 -- exact checks of every result in 02_Solutions.md.

Problem 1  Two cards from {2,...,6}: joint/marginal/conditional pmfs, moments,
           covariance, correlation coefficient, E{Y|X}
Problem 2  f_X(x) = 1/x^2 (x >= 1), Y | X=x ~ U[0,x]: joint pdf and marginal of Y
Problem 3  Uniform pdf on the triangle (0,0),(0,2),(2,1): a, conditionals, rho_XY

Run:  python tutorial2_verify.py      (needs numpy sympy matplotlib)
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)


def header(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------------------------------------------------------------------
header("Problem 1 - two cards from {2,...,6}")
pairs = list(combinations(range(2, 7), 2))
P = defaultdict(F)
for a, b in pairs:
    P[(a + b, abs(a - b))] += F(1, len(pairs))
xs = sorted({x for x, _ in P})
ys = sorted({y for _, y in P})

print("joint pmf P_XY(x,y)  (rows x, columns y):")
print("x\\y " + "".join(f"{y:>7}" for y in ys) + "   P_X(x)")
PX = {x: sum(P[(x, y)] for y in ys) for x in xs}
PY = {y: sum(P[(x, y)] for x in xs) for y in ys}
for x in xs:
    print(f"{x:>3} " + "".join(f"{str(P[(x, y)]):>7}" for y in ys) + f"   {str(PX[x]):>6}")
print("P_Y " + "".join(f"{str(PY[y]):>7}" for y in ys))

indep = all(P[(x, y)] == PX[x] * PY[y] for x in xs for y in ys)
print("independent?", indep, " e.g. P_XY(5,2) =", P[(5, 2)], "but P_X(5)P_Y(2) =", PX[5] * PY[2])

print("\nconditional pmf P_{Y|X}(y|x):")
for x in xs:
    row = {y: P[(x, y)] / PX[x] for y in ys if P[(x, y)] > 0}
    print(f"  x={x:>2}: ", {y: str(v) for y, v in row.items()})

E = lambda g: sum(g(x, y) * p for (x, y), p in P.items())
mX, mY = E(lambda x, y: x), E(lambda x, y: y)
vX = E(lambda x, y: (x - mX) ** 2)
vY = E(lambda x, y: (y - mY) ** 2)
rXY = E(lambda x, y: x * y)
cXY = rXY - mX * mY
print(f"\nmu_X={mX}  mu_Y={mY}  E{{X^2}}={E(lambda x,y:x*x)}  E{{Y^2}}={E(lambda x,y:y*y)}")
print(f"var X={vX}  var Y={vY}  correlation E{{XY}}={rXY}  covariance={cXY}  rho={cXY/(vX*vY)**0.5 if vX*vY else 'n/a'}")
condE = {x: sum(y * P[(x, y)] for y in ys) / PX[x] for x in xs}
print("E{Y|X=x}:", {x: str(v) for x, v in condE.items()})
print("tower check E{E{Y|X}} =", sum(condE[x] * PX[x] for x in xs))
assert (mX, mY, vX, vY, rXY, cXY) == (8, 2, 3, 1, 16, 0)

# ---------------------------------------------------------------------------
header("Problem 2 - f_X = 1/x^2 on [1,inf), Y|X=x ~ U[0,x]")
x, y = sp.symbols("x y", positive=True)
fXY = 1 / x**3  # on x >= 1, 0 <= y <= x
print("joint pdf f_XY = f_{Y|X} f_X =", fXY, " on {x>=1, 0<=y<=x}")
print("total mass =", sp.integrate(sp.integrate(fXY, (y, 0, x)), (x, 1, sp.oo)))
fY_low = sp.integrate(fXY, (x, 1, sp.oo))  # 0 <= y <= 1
fY_high = sp.integrate(fXY, (x, y, sp.oo))  # y > 1
print("f_Y(y) for 0<=y<=1 :", fY_low)
print("f_Y(y) for y>1     :", fY_high)
print("mass of f_Y:", sp.integrate(fY_low, (y, 0, 1)) + sp.integrate(fY_high, (y, 1, sp.oo)))
print("E{X} =", sp.integrate(x / x**2, (x, 1, sp.oo)), "(heavy tail: the mean does not exist)")
# Monte Carlo: X = 1/U (inverse cdf, F_X = 1 - 1/x), Y = X*V
U, V = rng.random(1_000_000), rng.random(1_000_000)
Xs = 1 / (1 - U)
Ys = Xs * V
print(f"simulation P(Y<=1) = {np.mean(Ys<=1):.4f} (exact 1/2);  P(Y>2) = {np.mean(Ys>2):.4f} (exact 1/4)")

# ---------------------------------------------------------------------------
header("Problem 3 - uniform on triangle (0,0),(0,2),(2,1)")
a = sp.symbols("a", positive=True)
area = sp.integrate(sp.integrate(1, (y, x / 2, 2 - x / 2)), (x, 0, 2))
a_val = sp.solve(sp.Eq(a * area, 1), a)[0]
print("area =", area, " => a =", a_val)
fx = sp.integrate(a_val, (y, x / 2, 2 - x / 2))
fy1 = sp.integrate(a_val, (x, 0, 2 * y))
fy2 = sp.integrate(a_val, (x, 0, 4 - 2 * y))
print("f_X(x) =", sp.simplify(fx), " on [0,2]")
print("f_Y(y) =", fy1, " on [0,1];  ", sp.expand(fy2), " on [1,2]")
print("f_{Y|X}(y|x) =", sp.simplify(a_val / fx), " on x/2 <= y <= 2 - x/2")
print("f_{X|Y}(x|y) =", sp.simplify(a_val / fy1), " (0<y<=1);  ", sp.simplify(a_val / fy2), " (1<=y<2)")


def EE(g):
    return sp.simplify(sp.integrate(sp.integrate(g * a_val, (y, x / 2, 2 - x / 2)), (x, 0, 2)))


mX, mY = EE(x), EE(y)
EX2, EY2, EXY = EE(x**2), EE(y**2), EE(x * y)
vX, vY = EX2 - mX**2, EY2 - mY**2
cov = EXY - mX * mY
print(f"E{{X}}={mX} E{{Y}}={mY} E{{X^2}}={EX2} E{{Y^2}}={EY2} E{{XY}}={EXY}")
print(f"var X={vX} var Y={vY} cov={cov}  rho={sp.simplify(cov/sp.sqrt(vX*vY))}")
assert a_val == sp.Rational(1, 2) and cov == 0

# ---------------------------------------------------------------------------
header("Figures")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Problem 1: joint pmf as a dot plot
fig, ax = plt.subplots(figsize=(6, 3.2))
for (xx, yy), p in [(k, v) for k, v in P.items() if v > 0]:
    ax.scatter(xx, yy, s=900 * float(p), color="#4c72b0")
    ax.annotate(str(p), (xx, yy), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
ax.plot(list(condE), [float(v) for v in condE.values()], "o--", color="#c44e52", label=r"$E\{Y|X=x\}$")
ax.axhline(2, color="gray", lw=0.8, ls=":", label=r"$\mu_Y=2$")
ax.set_xlabel("$x$ (sum)"); ax.set_ylabel("$y$ (|difference|)")
ax.set_ylim(0.3, 4.8); ax.set_xticks(xs); ax.set_yticks(ys)
ax.legend(loc="lower center", fontsize=8, ncol=2)
ax.set_title("Problem 1: joint pmf; dependent but uncorrelated")
fig.tight_layout(); fig.savefig(FIG / "p1_joint_pmf.png", dpi=150)

# Problem 2: joint support and marginal of Y
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
xg = np.linspace(1, 5, 200)
axs[0].fill_between(xg, 0, xg, color="#4c72b0", alpha=0.25)
axs[0].plot(xg, xg, color="#4c72b0")
axs[0].plot([1, 1], [0, 1], color="#4c72b0")
axs[0].set_xlim(0, 5); axs[0].set_ylim(0, 5)
axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$y$")
axs[0].set_title(r"support of $f_{XY}=1/x^3$: $x\geq1,\ 0\leq y\leq x$", fontsize=9)
yg = np.linspace(0, 5, 500)
fyg = np.where(yg <= 1, 0.5, 1 / (2 * np.maximum(yg, 1e-9) ** 2))
sel = Ys < 5  # show [0,5) but normalise by ALL samples so the bars estimate f_Y itself
axs[1].hist(Ys[sel], bins=100, range=(0, 5), weights=np.full(sel.sum(), 1 / (len(Ys) * 0.05)),
            color="#9aa5b1", label="simulation")
axs[1].plot(yg, fyg, color="#c44e52", label="$f_Y(y)$")
axs[1].set_xlabel("$y$"); axs[1].legend(fontsize=8)
axs[1].set_title("marginal of $Y$", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "p2_support_marginal.png", dpi=150)

# Problem 3: the triangle and conditional slices
fig, ax = plt.subplots(figsize=(4.6, 3.8))
tri = np.array([[0, 0], [2, 1], [0, 2], [0, 0]])
ax.fill(tri[:, 0], tri[:, 1], color="#4c72b0", alpha=0.2)
ax.plot(tri[:, 0], tri[:, 1], color="#4c72b0")
x0 = 0.8
ax.plot([x0, x0], [x0 / 2, 2 - x0 / 2], color="#c44e52", lw=2.5, label=r"slice $X=0.8$: $Y\sim U[0.4,1.6]$")
y0 = 0.6
ax.plot([0, 2 * y0], [y0, y0], color="#55a868", lw=2.5, label=r"slice $Y=0.6$: $X\sim U[0,1.2]$")
ax.axhline(1, color="gray", lw=0.8, ls=":")
ax.text(1.35, 1.6, "$y = 2 - x/2$", fontsize=9); ax.text(1.35, 0.45, "$y = x/2$", fontsize=9)
ax.set_xlim(-0.1, 2.3); ax.set_ylim(-0.1, 2.3); ax.set_aspect("equal")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(fontsize=7, loc="upper right")
ax.set_title("Problem 3: region $R$, $a=1/2$", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "p3_triangle.png", dpi=150)
print("saved figures to", FIG)
