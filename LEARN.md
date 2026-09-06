# Local Volatility: From the Implied Surface to Dupire, PDE and Monte Carlo

This document teaches the theory behind the `localvol` project: what local
volatility is, how Dupire's equation turns an implied-vol surface into a
diffusion, how to build the surface without introducing arbitrage, how the
finite-difference and Monte Carlo pricers work, and where the model breaks
down in real markets. Formulas use the project's notation throughout:
spot $S_0$, rates $(r, q)$ (equity) or $(r_d, r_f)$ (FX), forward
$F(T) = S_0 e^{(r-q)T}$, forward log-moneyness $k = \ln(K/F(T))$, and total
implied variance $w(k, T) = \sigma_{\mathrm{imp}}^2(k, T)\, T$.

---

## 1. Three kinds of volatility

**Implied volatility** is not a model — it is a *quoting convention*. Given a
vanilla option's market price, $\sigma_{\mathrm{imp}}(K, T)$ is the number
you must feed into Black-Scholes to reproduce that price. The market's
disagreement with Black-Scholes shows up as the *smile*: implied vol varies
with strike and expiry, which flat-vol Black-Scholes cannot produce. The
implied surface is a static snapshot of vanilla prices, nothing more; it says
nothing by itself about how to price a barrier or a forward-start.

**Local volatility** (Dupire 1994, Derman-Kani 1994) is the *minimal* model
that upgrades the snapshot to a full dynamic: keep a one-factor diffusion,
but let the instantaneous volatility be a deterministic function of spot and
time,

$$
\frac{dS_t}{S_t} = (r - q)\, dt + \sigma_{\mathrm{loc}}(S_t, t)\, dW_t .
$$

Dupire's remarkable result is that there is *exactly one* such function
consistent with today's entire vanilla surface — the model calibrates to
every quoted strike and expiry by construction. Once you have
$\sigma_{\mathrm{loc}}$, any payoff (American, barrier, Asian) can be priced
by PDE or Monte Carlo *consistently with all vanillas at once*.

**Stochastic volatility** (Heston, SABR, ...) makes volatility its own random
process with its own driver, correlated with spot. It cannot generally match
the vanilla surface exactly with a handful of parameters, but its *dynamics*
are far more realistic: smiles move with the market, forward smiles do not
flatten, and vol-of-vol is a genuine risk factor. The practical synthesis is
**local-stochastic volatility (LSV)**: a stochastic-vol backbone multiplied
by a local "leverage" function calibrated so vanillas are repriced exactly —
the standard model on FX exotics desks.

A useful way to remember the hierarchy: implied vol is *what the market
says*, local vol is *the unique smile-consistent one-factor story*, and
stochastic vol is *a more honest story that no longer fits the smile
exactly*. Gyöngy's theorem makes the link precise: for any process with
stochastic volatility, the local-vol function

$$
\sigma_{\mathrm{loc}}^2(K, T) = \mathbb{E}\!\left[\sigma_t^2 \,\middle|\, S_T = K\right]
$$

defines a Markovian one-factor diffusion with the *same marginal
distributions* — hence the same vanilla prices — but generally different
joint distributions, hence different exotics prices. That single sentence
explains both why local vol reprices vanillas perfectly and why it can
misprice path-dependent products.

---

## 2. Dupire's equation — derivation sketch

### 2.1 The price-space form

Two ingredients:

1. **Breeden-Litzenberger**: the undiscounted risk-neutral density of $S_T$
   is the second strike derivative of call prices,
   $\varphi_T(K) = e^{rT} \partial^2 C / \partial K^2$.
2. **Fokker-Planck**: under the local-vol SDE, that density satisfies the
   forward Kolmogorov equation in $(K, T)$.

Integrate the Fokker-Planck equation twice against the call payoff
$(S-K)^+$ (two integrations by parts, boundary terms vanishing for a
well-behaved density) and you obtain **Dupire's forward equation** for call
prices as functions of strike and expiry, today's date fixed:

$$
\frac{\partial C}{\partial T}
 = \tfrac{1}{2}\, \sigma_{\mathrm{loc}}^2(K, T)\, K^2 \frac{\partial^2 C}{\partial K^2}
 - (r - q)\, K \frac{\partial C}{\partial K} - q\, C .
$$

Solved for the local variance:

$$
\sigma_{\mathrm{loc}}^2(K, T)
 = \frac{\dfrac{\partial C}{\partial T} + (r-q) K \dfrac{\partial C}{\partial K} + q C}
        {\tfrac{1}{2} K^2 \dfrac{\partial^2 C}{\partial K^2}} .
$$

Note the direction: the ordinary Black-Scholes PDE runs *backward* in $(S, t)$
for one contract; Dupire's runs *forward* in $(K, T)$ across all contracts.
The numerator is (up to carry terms) the calendar spread; the denominator is
the butterfly. Positivity of both is precisely the absence of calendar and
butterfly arbitrage — Dupire's formula is well defined exactly when the
surface is arbitrage-free.

### 2.2 Why nobody computes it that way: the total-variance form

Differentiating market *prices* is numerically brutal: deep in- or
out-of-the-money, $\partial^2 C/\partial K^2$ is a tiny number obtained by
cancelling large ones, and the ratio of two noisy small numbers is garbage.
The fix (Gatheral) is to change variables to forward log-moneyness
$k = \ln(K/F(T))$ and total implied variance $w(k, T)$, substitute the
Black-Scholes formula $C = C_{BS}(k, w)$, and let the chain rule absorb all
the near-cancellations analytically. The result — the formula this project
implements — is

$$
\sigma_{\mathrm{loc}}^2(k, T) =
\frac{\dfrac{\partial w}{\partial T}}
     {1 - \dfrac{k}{w}\dfrac{\partial w}{\partial k}
        + \dfrac{1}{4}\left(-\dfrac{1}{4} - \dfrac{1}{w} + \dfrac{k^2}{w^2}\right)
          \left(\dfrac{\partial w}{\partial k}\right)^{2}
        + \dfrac{1}{2}\dfrac{\partial^2 w}{\partial k^2}} .
$$

Why this is numerically better:

* $w$ is a **smooth, slowly varying, dimensionless** function of $k$ —
  typically a gentle parabola-like smile — whereas $C(K)$ ranges over many
  orders of magnitude across the same strikes. Finite differences of $w$ are
  well conditioned everywhere the smile is quoted.
* The dangerous cancellations (vega factors, discount factors, the
  lognormal density) are performed *symbolically* by the change of
  variables, not numerically.
* The structure is transparent: the **numerator** $\partial w/\partial T$ is
  the forward variance — positivity is *calendar* no-arbitrage. The
  **denominator** is $g(k)$, proportional to the implied risk-neutral
  density — positivity is *butterfly* no-arbitrage. When a clamp fires in
  the code you know exactly which arbitrage the surface is flirting with.
* Rates and dividends drop out entirely: $w(k, T)$ is carry-independent, so
  one surface file serves equity and FX; the pricer reintroduces carry
  through its own forward when converting levels to $k$.

**Sanity anchor.** For a flat surface $w(k, T) = \sigma^2 T$:
$\partial w/\partial T = \sigma^2$, both $k$-derivatives vanish, the
denominator is $1$, so $\sigma_{\mathrm{loc}} \equiv \sigma$. The project's
`flat_dupire_*` golden cases assert this to $10^{-6}$ — a strong end-to-end
test of the surface interpolation, the finite differences and the formula in
one shot.

### 2.3 Worked example (ATM, $T = 1$; golden `dupire_bundled_k100_T1001`)

The bundled surface is an SSVI family with $\theta_T = 0.04\,T$,
$\rho = -0.5$, $\eta = 0.7$, $\gamma = 0.5$; at the money it has
$w(0, T) = \theta_T$ exactly, i.e. a flat 20% ATM term structure. Querying
the interpolated surface at $k = 0$, $T = 1$ with the contractual steps
$\Delta k = 10^{-3}$, $\Delta T = 10^{-4}$:

| Quantity | Value |
|---|---|
| $w$ | $0.0400000$ |
| $\partial w/\partial k$ (central) | $-0.0686316$ |
| $\partial^2 w/\partial k^2$ (central) | $+0.1759576$ |
| $\partial w/\partial T$ (central) | $+0.0400000$ |
| denominator $g$ | $1.0582450$ |
| $\sigma_{\mathrm{loc}} = \sqrt{0.04/1.05825}$ | $\mathbf{0.19441818}$ |

The golden case is evaluated at $T = 1.001$ rather than exactly on the
pillar (value `0.1944178773`; the difference from the $T = 1$ number above
is $3 \times 10^{-7}$ because ATM total variance is linear in $T$). The
reason is §3: linear-in-$w$ time interpolation makes $\partial w/\partial T$
piecewise constant, so local vol *jumps* at every pillar — at $k = \ln 0.8$
it reads $0.3054 \mid 0.2985 \mid 0.2914$ across $T = 0.5 \pm 3\times10^{-4}$
— and a value sampled exactly on the pillar is the average of two one-sided
limits, which a port that differentiates one-sidedly would not reproduce.

Notice $\sigma_{\mathrm{loc}}(0, 1) < \sigma_{\mathrm{imp}}(0, 1) = 0.20$
even though ATM total variance grows exactly like a 20%-vol process: with a
negative skew the denominator exceeds 1 at $k=0$, i.e. the implied density
piles extra mass there. Local vol is a *redistribution* of the implied
variance across the $(k, T)$ plane, not a relabeling.

**The "two times" rule.** Expanding the formula for small $T$ and gentle
skew gives the classic result that the local-vol skew is about **twice** the
implied skew,
$\partial_k \sigma_{\mathrm{loc}} \approx 2\, \partial_k \sigma_{\mathrm{imp}}$
(intuition: the implied vol at $k$ is roughly the *average* of local vol
along the path from spot to strike, so local vol must fall twice as fast for
the average to fall as observed). On the bundled surface at $T = 1$ the
measured ratio is $1.68$ — the rule is exact only in the short-expiry,
linear-skew limit; curvature and maturity erode it.

### 2.4 The short-expiry limit ($T \to 0$)

At $T = 0$ the surface has $w = 0$ and the formula reads $0/0$. The limit is
nevertheless finite and explicit. Write $w(k, T) = s(k)^2 T$ for small $T$
(flat forward variance below the first pillar, which is exactly the
project's extrapolation rule) with $s(k) = \sigma_{\mathrm{imp}}(k, 0)$.
Then $\partial_T w = s^2$, $\partial_k w = 2 s s' T$,
$\partial_{kk} w = 2 (s'^2 + s s'') T$, and in the denominator

$$
\frac{k}{w}\,\partial_k w = \frac{2 k s'}{s}, \qquad
\frac{1}{w}(\partial_k w)^2 = 4 s'^2 T \to 0, \qquad
\frac{k^2}{w^2}(\partial_k w)^2 = \frac{4 k^2 s'^2}{s^2},
$$

while $\tfrac{1}{2}\partial_{kk} w \to 0$. The denominator therefore tends to
$1 - 2 k s'/s + k^2 s'^2/s^2 = (1 - k s'/s)^2$, and

$$
\sigma_{\mathrm{loc}}(k, 0) = \frac{s(k)}{1 - k\, s'(k)/s(k)}
= \frac{\sigma_{\mathrm{imp}}}{1 - k\, \partial_k \ln \sigma_{\mathrm{imp}}}
$$

— the Berestycki–Busca–Florent (2002) result, whose inverse form
"$1/\sigma_{\mathrm{imp}}(k,0)$ is the harmonic mean of $1/\sigma_{\mathrm{loc}}$
between $0$ and $k$" is the rigorous version of the "two times" rule. The
code uses this formula *verbatim* for `vol(k, 0)` (with $s'$ by the same
central $\Delta k$ difference) so that the local-vol surface is continuous
at $T = 0$; on the bundled surface `vol(-0.3, 0) = 0.4524`, which the
$T > 0$ branch approaches as $T \to 0$ (an earlier revision returned the
*implied* vol $0.309$ there — a 45% error for any user dumping a
spot–vol grid at $t = 0$). Two consequences worth knowing: the denominator
$1 - k s'/s$ is the intercept at $k = 0$ of the tangent to $s$ at $k$, so a
convex smile can never make it negative, while a strongly convex parabola
such as $s = 0.05 + 4k^2$ does at $|k| = 0.5$ — the `T = 0` branch then caps
and counts, exactly like the $T > 0$ branch.

---

## 3. Building the surface: interpolation, extrapolation, arbitrage

Dupire differentiates whatever you hand it — including your interpolator's
artifacts. Surface construction *is* the hard part of local vol in practice.

**Choices made in this project (and why):**

* **Store total variance, not vol.** Calendar arbitrage becomes
  monotonicity of $w$ in $T$ at fixed $k$; linear interpolation *in $w$*
  between pillars preserves that monotonicity automatically. Interpolating
  implied vol linearly in $T$ does not.
* **Natural cubic spline in $k$ per expiry.** Dupire needs
  $\partial^2 w/\partial k^2$, so the interpolant must be $C^2$; a cubic
  spline is the simplest $C^2$ interpolant fully determined by the nodes.
  "Natural" boundary conditions set $w'' = 0$ at the end nodes — but
  *not* $w' = 0$, so the transition into the flat wings is only $C^0$: a
  genuine kink in slope at the last quoted strike (see the extrapolation
  bullet). The spline's tridiagonal moment system is solved with the same
  Thomas kernel the PDE uses, so all four language ports reproduce the
  surface bit-comparably. Beware: splines can *overshoot* on ragged data,
  manufacturing spurious convexity or even $w < 0$ — the surface scans
  every node interval at build time and reports such dips in
  `negative_w_count`; acceptable here because the bundled data is smooth by
  construction, but on real quotes desks fit a parametric form (SVI/SSVI)
  per slice instead, precisely to control this.
* **Linear-in-$w$ in $T$ between pillars** (see above), giving a piecewise
  constant forward variance $\partial w/\partial T$ between pillars.
* **Extrapolation** — always a modeling *choice*, never data:
  * $k$ beyond the quoted wings: clamp $k$ (flat total variance ⇒ flat vol
    wings). Simple and safe-ish, but $w$ is only $C^0$ at the boundary. A
    central finite-difference stencil that straddles that kink reads
    $\partial_{kk} w \approx -w'(k_{\max})/\Delta k \approx -50$, the
    denominator turns negative and the vol is capped at 500% — **at the
    quoted wing**, 2.5 standard deviations from ATM at $T = 1$, well inside
    the PDE grid. The fix is to clamp the *query point* into
    $[k_{\min} + \Delta k,\; k_{\max} - \Delta k]$ before differencing and
    to use the clamped $k$ in every term: local vol is then constant in $k$
    beyond $k_{\max} - \Delta k$, continuous across the wing, and no clamp
    fires anywhere on the bundled surface. Before this clamp the demo's
    round-trip error was 16.4 bp, about 85% of it caused by those spikes;
    after it, 2.7 bp.
  * $T$ below the first pillar: $w(k, T) = w(k, T_1)\, T / T_1$ — flat
    forward variance, i.e. constant implied vol down to $T = 0$.
  * $T$ beyond the last pillar: continue the last interval's slope in $w$,
    floored at zero so $w$ never decreases (no manufactured calendar
    arbitrage).
* **Single expiry**: the flat-forward rule is applied on both sides and a
  warning is emitted — $\partial w/\partial T$ then rests on an assumption,
  not on data.

**Arbitrage handling.** At build time the surface counts every node pair
with $w(k_i, T_{j+1}) < w(k_i, T_j) - 10^{-12}$ (`calendar_violations`) and
warns; it does not repair or throw — the caller decides. At evaluation time
Dupire clamps: numerator $\le 0$ (calendar arb / no forward variance) floors
the vol at 1%, denominator $\le 0$ (butterfly arb, typically wing artifacts)
caps it at 500%, and both counters are reported. The philosophy is
*detect and report, never silently repair, never crash mid-pricing*.

---

## 4. Pricing PDE: finite differences done carefully

### 4.1 The equation and the grid

In log-spot $x = \ln S$ with time-to-maturity $\tau = T - t$, the backward
pricing PDE under local vol reads

$$
\frac{\partial V}{\partial \tau}
 = \underbrace{\tfrac{1}{2}\sigma^2(x, t)}_{a}\, \frac{\partial^2 V}{\partial x^2}
 + \underbrace{\left(r - q - \tfrac{1}{2}\sigma^2\right)}_{\mu} \frac{\partial V}{\partial x}
 - r V .
$$

Log-spot makes the coefficients state-independent for flat vol and keeps the
grid uniform — and a uniform grid is what makes the Thomas solver and the
convergence analysis clean. The grid is
$x_i = \ln S_0 + (i - M/2) h$, $i = 0..M$, with half-width
$W = |\ln(K/S_0)| + n_{sd}\,\sigma_{\mathrm{ref}}\sqrt{T} + |r - q|\,T$
(default 6 standard deviations plus drift displacement, strike always
covered). Because $\ln S_0$ is *exactly* the middle node, the price is read
off with no interpolation error.

### 4.2 Explicit, implicit, Crank-Nicolson — the theta scheme

Discretize space with central differences and write one time step as

$$
\left(I - \theta\, \Delta\tau\, L\right) V^{n+1}
 = \left(I + (1-\theta)\, \Delta\tau\, L\right) V^{n},
$$

where $L$ is the tridiagonal operator with rows
$(\alpha_i - \beta_i,\; -2\alpha_i - r,\; \alpha_i + \beta_i)$,
$\alpha_i = a_i/h^2$, $\beta_i = \mu_i/(2h)$.

* $\theta = 0$ — **explicit** (forward Euler): no linear solve, but only
  conditionally stable — it requires roughly
  $\Delta\tau \le h^2/(2a)$, which for fine spatial grids forces absurdly
  many time steps. First-order in time.
* $\theta = 1$ — **implicit** (backward Euler): unconditionally stable and
  strongly damping (L-stable: the amplification factor of stiff modes tends
  to 0), but only first-order in time.
* $\theta = 1/2$ — **Crank-Nicolson**: unconditionally stable *and*
  second-order in both $\Delta\tau$ and $h$. The workhorse — with one flaw.

### 4.3 The CN flaw and Rannacher smoothing

CN is A-stable but **not L-stable**: for a stiff Fourier mode with
eigenvalue $\lambda \ll 0$ the amplification factor
$(1 + \tfrac{1}{2}\Delta\tau\lambda)/(1 - \tfrac{1}{2}\Delta\tau\lambda)
\to -1$. Such modes are not amplified — the scheme never blows up — but they
are barely damped and flip sign every step. A vanilla payoff has a *kink* at
the strike (a barrier payoff a jump), which loads exactly those
high-frequency modes; the symptom is a sawtooth superimposed on the price
near the strike, far worse in gamma, decaying only slowly. Making
$\Delta\tau$ large relative to $h$ ("dt too big vs dx") makes it vivid even
though nothing is unstable.

**Rannacher's fix**, as specified for all ports: replace the *first* time
interval by **two backward-Euler half-steps** of size $\Delta\tau/2$, then
run $N-1$ ordinary CN steps. Backward Euler's amplification factor tends to
0, so the two implicit half-steps annihilate precisely the kink-excited
stiff modes; because only an $O(\Delta\tau)$-long initial segment is
sub-second-order, the scheme's global second-order accuracy survives. The
test suite checks both effects: gamma near the strike is non-oscillatory,
and the observed convergence order on grid-halving lies in $[1.5, 2.5]$
(measured $\approx 2$). A literature note: Giles and Carter (2006) show
that *four* backward-Euler half-steps (two $\Delta\tau$ intervals) are
needed for delta and gamma at the kink to converge at second order as well;
with the two used here the price is second order but the Greeks retain an
$O(\Delta\tau)$ component. This is a documented choice — the goldens pin
prices, not Greeks — and the schedule is a one-line change in each port.

### 4.4 Boundary conditions and the linear solve

Dirichlet boundaries from discounted asymptotics at the new time level: a
call is worthless at $x_0$ and worth
$S_M e^{-q\tau} - K e^{-r\tau}$ (floored at 0) at $x_M$; mirrored for puts.
The boundary values are folded into the first/last rows of the right-hand
side, and each step is one tridiagonal solve by the **Thomas algorithm**
(Gaussian elimination without pivoting, $O(M)$) — safe because the
matrices are diagonally dominant M-matrices whenever the mesh Péclet
condition $|\mu_i| h \le 2 a_i$ holds at every node. That is *not*
automatic: the width rule uses $\sigma_{\mathrm{ref}}$ while each node uses
its own $\sigma_i$, so a node floored at 1% by the Dupire clamp with a 1–5%
carry violates it (so does a 1% flat vol with a 10% carry). With central
differencing the lower coefficient $\alpha_i - \beta_i$ then turns negative
and the scheme loses monotonicity — a measured example: a 1%-vol put with
$r = 10\%$ on a $100\times50$ grid comes out at $-3\times10^{-6}$ with grid
values down to $-0.08$. The code therefore switches, node by node, to
first-order **upwind** differencing of the drift wherever the condition
fails ($\text{lower}_i = \alpha_i + \max(-\mu_i, 0)/h$,
$\text{upper}_i = \alpha_i + \max(\mu_i, 0)/h$,
$\text{center}_i = -2\alpha_i - |\mu_i|/h - r$): both off-diagonals stay
non-negative, the row sum stays $-r$, prices stay non-negative and
monotone, at the price of $O(|\mu| h/2)$ numerical diffusion at those (by
construction already pathological) nodes. Nodes that satisfy the condition
are untouched, so flat-vol results on the default grids are bit-identical
to plain central differencing. No library banded solver appears in the
pricing path in any language; SciPy / Eigen are used only as independent
cross-checks inside tests.

Under local vol, the coefficient $\sigma_i$ for the step from $\tau$ to
$\tau + \Delta\tau_s$ is frozen at the step's **midpoint calendar time** and
looked up at forward log-moneyness $k_i = x_i - \ln F(t_{\mathrm{mid}})$ —
the same coordinates the Dupire surface is built in, so PDE and MC
discretize the identical diffusion.

### 4.5 American options: PSOR

Early exercise turns each time step into a **linear complementarity
problem**: find $V \ge \psi$ (obstacle $\psi_i = (K - S_i)^+$) with
$AV \ge b$ and $(AV - b)^\top (V - \psi) = 0$. This project solves it by
**PSOR** — projected successive over-relaxation: sweep the nodes in order,
compute the Gauss-Seidel update, over-relax with $\omega = 1.5$, and project
onto the obstacle:

$$
V_i \leftarrow \max\!\left(\psi_i,\; V_i + \omega\, (V_i^{GS} - V_i)\right),
$$

iterating until the sup-norm update falls below $10^{-8}$. PSOR was chosen
over a penalty / operator-splitting scheme because it solves the LCP to an
explicit tolerance with **no penalty parameter to tune**, and warm-starting
each step from the previous time level (projected onto the obstacle) keeps
iteration counts small on these grids. Non-convergence at `max_iter` is
reported via a warning and the computation proceeds — consistent with the
project's "report, don't crash" policy. Golden anchor: the flat-vol American
put (S=K=100, r=5%, σ=20%, T=1) must match a 5000-step CRR binomial value
`6.0902194081` within 1e-2; the European put is worth 5.5735, so the
early-exercise premium is ≈ 0.5167 — always $\ge 0$, as a test asserts.

---

## 5. Local-vol Monte Carlo

### 5.1 Scheme and bias

Simulate $X = \ln S$ on a uniform grid with **log-Euler**:

$$
X_{n+1} = X_n + \left(r - q - \tfrac{1}{2}\sigma_n^2\right)\Delta t
 + \sigma_n \sqrt{\Delta t}\, Z_n,
\qquad
\sigma_n = \sigma_{\mathrm{loc}}\!\left(X_n - \ln F(t_n),\, t_n\right),
$$

with the vol looked up at the **start** of each step. Euler on $\ln S$
rather than on $S$ keeps $S > 0$ exactly and behaves better for lognormal-ish
dynamics. Two distinct errors matter:

* **Discretization (weak) bias**, $O(\Delta t)$: the frozen-vol step
  distribution is not the true one when $\sigma_{\mathrm{loc}}$ varies along
  the step. It shrinks with more steps, not more paths.
* **Statistical error**, $O(1/\sqrt{n_{\mathrm{paths}}})$: reported as a
  standard error. **Antithetic variates** (default): paths come in $\pm Z$
  pairs; each mirrored path recomputes its own vol from its own state. The
  estimator averages each pair *first* and computes the standard error over
  the pair means — computing it over raw paths would overstate the error,
  since paired payoffs are negatively correlated.

Cross-checks are statistical by design: European MC must sit within 3
standard errors of the PDE price, and the golden MC tolerances were sized at
4 SE so that each language can use its own fixed-seed generator yet still be
falsifiable.

### 5.2 Barriers and the Brownian bridge

Naive discrete monitoring — kill a path when a *grid point* lands at or
above $b = \ln B$ — systematically **overprices** an up-and-out option: the
path may have crossed the barrier *between* grid points and come back,
unseen. The bias decays only like $O(\sqrt{\Delta t})$, painfully slowly.

The **Brownian-bridge correction** removes the leading bias without extra
random numbers. Conditional on step endpoints $x_0, x_1 < b$ with step vol
$\sigma$, the probability that the bridge between them touched the barrier
is exactly

$$
p_{\mathrm{cross}} = \exp\!\left(-\frac{2 (b - x_0)(b - x_1)}{\sigma^2 \Delta t}\right),
$$

so each surviving path carries a multiplicative survival weight
$\prod_n (1 - p_{\mathrm{cross},n})$ and the payoff is weighted accordingly.
The corrected price is always $\le$ the discrete-monitoring price, and both
must be $\le$ the vanilla (asserted in tests). The golden barrier case
checks the bridged MC against the continuous-barrier analytic
(Reiner-Rubinstein) value `3.1288387727` — versus a vanilla call worth
8.9160 with the same parameters; the knock-out feature destroys almost two
thirds of the value.

Under local vol the "bridge vol" is the frozen step vol — a further
approximation, which is why the golden tolerance carries a small
residual-bias allowance on top of 4 SE.

---

## 6. Where local volatility breaks: dynamics

Local vol matches today's vanilla surface *perfectly* and today's smile
*statically*. Its failures are all about how the smile **moves**:

* **Forward smile flattening.** The model's implied smile for a forward
  start (an option struck ATM at future $T_1$, expiring at $T_2$) is far
  flatter than the spot smile — as calendar time passes, the model rides
  into the flatter far region of $\sigma_{\mathrm{loc}}(S, t)$, and
  conditional on $S_{T_1}$ the relevant local-vol patch looks nearly flat.
  Real markets re-quote a full smile every day. Consequence: local vol
  **underprices cliquets and forward-start smile risk**; never use it for
  products keyed on the *future* smile.
* **Smile dynamics / hedging.** Empirically, when spot falls the whole
  equity smile tends to shift (sticky-delta-ish in FX, somewhere between
  sticky-strike and sticky-delta in equities). Local vol implies its own
  rigid "sticky-local" dynamic in which the smile moves *opposite* to spot
  more than observed, so local-vol deltas and vegas differ systematically
  from Black-Scholes-with-market-smile hedges — famously criticized by
  Hagan et al. (the SABR paper) for producing wrong-way smile deltas.
* **No vol-of-vol.** Volatility is a deterministic function of spot: spot
  and vol are perfectly (anti-)correlated, and there is no independent
  volatility risk. Products convex in volatility (options on realized
  variance, cliquets, some barriers) are mispriced.
* **Barrier prices sit at one end.** For many barrier payoffs, pure local
  vol and pure stochastic vol bracket the market; FX desks calibrate a
  **mixing weight** in LSV models to interpolate. Pure Dupire is the
  0%-stochastic corner.

**Desk practice.**

* *Equity*: local vol is the default engine for vanilla books, light
  exotics, autocallables (often with discrete dividends bolted on — cash
  dividends break the clean Dupire formula and need care), and as the
  calibration target of LSV. Quotes come as strike/expiry grids or
  moneyness-parametrized surfaces; SVI/SSVI slice fits are the standard
  arbitrage-free smoothers feeding Dupire.
* *FX*: quotes arrive as ATM / 25Δ risk-reversal / 25Δ butterfly per tenor
  and must be converted to strikes via the appropriate delta convention
  before a surface is built. Garman-Kohlhagen means every equity formula
  applies with $r = r_d$, $q = r_f$ — this project's `Market.fx` makes that
  explicit. Barriers and touches dominate the FX exotics flow and are priced
  on LSV, with pure local vol as a limiting/reference case.

---

## 7. Common pitfalls and numerical issues

1. **Differentiating noise.** Dupire's denominator involves second
   derivatives; any interpolation kink or bid-ask noise is amplified.
   Always differentiate a *smoothed* (ideally parametric, arbitrage-checked)
   surface — never raw quotes.
2. **Wing extrapolation artifacts.** Flat-$k$ extrapolation creates a
   slope kink at the last quoted strike. A finite-difference stencil that
   straddles it manufactures a huge negative curvature and caps the local
   vol at 500% *at the quoted wing* — not "far out in the grid": on the
   bundled surface that is 2.5 sd from ATM at $T = 1$ and 1.8 sd at
   $T = 2$, and it cost 13 of the demo's former 16 bp. Clamp the stencil
   inside the quoted box (as this project now does) and the counters read
   zero; after that, *any* clamp on a smooth surface means the input has
   real butterfly or calendar arbitrage.
3. **Vol-interpolation in the wrong variable.** Linear-in-vol time
   interpolation can manufacture calendar arbitrage between pillars even
   when the pillars themselves are clean. Interpolate total variance.
4. **CN without Rannacher.** Prices look fine; *gamma* is a sawtooth. Any
   scheme change must re-verify the damping test, not just price accuracy.
5. **Reading the price off-grid.** Interpolating the final grid at $S_0$
   adds an avoidable $O(h^2)$-with-bad-constant error; build the grid so
   $\ln S_0$ is a node (this project pins it to the exact middle node).
6. **MC vol lookup coordinates.** The surface is in *forward*
   log-moneyness: the lookup is $k = X_n - \ln F(t_n)$, not
   $X_n - \ln S_0$. Getting this wrong shows up only with nonzero carry —
   test with $r \ne q$ (the suites do: $r = 5\%, q = 2\%$ and an FX case
   with a negative foreign rate; the $\ln S_0$ bug lands 4–5 standard errors
   from the PDE).
7. **Antithetic standard errors.** Use pair means. Raw-path standard errors
   are biased high (the test suite would catch the resulting inconsistency).
8. **Statistical golden tests.** Cross-language MC cannot match
   bit-for-bit with different generators; size tolerances in standard errors
   (4 SE here) and keep seeds fixed per language so failures are
   reproducible.
9. **Forgetting the drift correction.** Log-Euler needs the
   $-\tfrac{1}{2}\sigma_n^2 \Delta t$ term with the *per-step, per-path*
   $\sigma_n$; using a constant correction silently biases local-vol MC.
10. **PSOR relaxation.** $\omega$ must lie in $(0, 2)$; $\omega = 1.5$ is a
    good default on these grids, but PSOR convergence degrades on very fine
    grids — monitor the non-convergence warnings before trusting American
    prices from unusual grid settings.
11. **The $T = 0$ slice.** $w(k, 0) = 0$ makes the formula $0/0$; returning
    the implied vol there (as one might by reflex) is wrong off-ATM by
    exactly the factor $1 - k\,\partial_k \ln\sigma_{\mathrm{imp}}$ (§2.4).
    Define the slice as the short-time limit and test continuity against
    $T = 10^{-6}$.
12. **Silent NaN.** A user vol callable that returns NaN, $\infty$ or a
    negative number must be rejected at the lookup, not discovered as a NaN
    price or a mysterious solver failure two layers up. Same for the
    antithetic standard error with a single pair ($0/0$): validate
    `n_paths` before simulating.

### 7.1 How a desk would use this

A realistic workflow around this code base, and where its boundaries lie:

1. **Quotes → surface.** Collect vanilla quotes per expiry (equity:
   strike/expiry grids; FX: ATM / 25Δ risk-reversal / 25Δ butterfly
   converted to strikes under the correct delta convention), fit an
   arbitrage-free parametric slice (SVI/SSVI) per expiry, and *sample* the
   fit onto a rectangular $(T, k)$ grid in forward log-moneyness — exactly
   the format of `data/implied_surface.csv`. Do not feed raw quotes to the
   spline. Convert dates to year fractions with one convention and use the
   same one for the pricer inputs; the library carries no calendar.
2. **Build and inspect.** Load the CSV, then read `calendar_violations`
   and `negative_w_count`. Both should be zero; if not, go back to the fit.
   Build `DupireLocalVol` once per surface and share it (it is immutable and
   its counters are thread-safe), reset the counters, and run a coarse
   $(k, t)$ sweep to confirm the floor/cap counters stay at zero *inside the
   quoted box* — a non-zero count there is butterfly/calendar arbitrage in
   the input, not a numerical artifact.
3. **Reprice the inputs (round trip).** Before pricing anything exotic,
   reprice the quoted vanillas by PDE and invert: the demo's grid should
   come back within a few basis points (2.7 bp here). This is the model's
   own consistency check and the cheapest way to catch a bad slice fit.
4. **Price.** Europeans and American puts by PDE (a few ms each natively),
   barriers by MC with the bridge correction; read the clamp counters again
   afterwards. For a strike grid, price in parallel on the shared object.
5. **Know what you are not getting.** No Greeks beyond the value grid
   (bump-and-reprice with a *re-derived* local vol is the non-trivial
   part), no discrete dividends or borrow curves, no stochastic component
   — forward-start and cliquet-style payoffs are mispriced by construction
   (§6), and barrier prices sit at the pure-local-vol end of the LSV range.
   Local vol here is the calibration target and reference case, not the
   final exotic model.

---

## 8. Interview-style Q&A

**Q1. What is the difference between implied, local and stochastic
volatility?**
Implied vol is a quoting convention — the Black-Scholes vol reproducing one
vanilla price. Local vol is the unique deterministic function
$\sigma(S, t)$ making a one-factor diffusion consistent with *all* vanilla
prices (Dupire). Stochastic vol gives volatility its own random driver:
better dynamics, but no longer an exact fit to the surface. By Gyöngy's
theorem, $\sigma_{\mathrm{loc}}^2(K,T) = \mathbb{E}[\sigma_t^2 | S_T = K]$
links the two: same marginals, different path behavior.

**Q2. Derive the sign conditions hiding in Dupire's formula.**
In total-variance form the numerator is $\partial w/\partial T$ — forward
variance, positive iff no calendar arbitrage. The denominator $g(k)$ is
proportional to the risk-neutral density — positive iff no butterfly
arbitrage. A negative numerator/denominator is not a numerical accident; it
is an arbitrage in the input surface (or an interpolation artifact), which
is why this implementation clamps and *counts* rather than throwing.

**Q3. Why is the total-variance form preferred to the call-price form
numerically?**
Because $w(k, T)$ is smooth, dimensionless and of order one across the whole
surface, while $\partial^2 C/\partial K^2$ is a near-cancellation of numbers
spanning many orders of magnitude in the wings. The change of variables does
the dangerous cancellations analytically and additionally removes all
rate/dividend dependence from the surface object.

**Q4. Flat implied surface at 20% — what is the local vol, and why is that
a good test?**
Exactly 20% everywhere: $\partial w/\partial T = \sigma^2$, the
$k$-derivatives vanish, the denominator is 1. It exercises the entire
pipeline (parsing, interpolation, finite differences, formula) and any
deviation beyond ~1e-6 flags an interpolation or differencing bug — hence
golden cases at tolerance $10^{-6}$.

**Q5. Why does Crank-Nicolson oscillate for option payoffs, and what is
the standard fix?**
CN is A-stable but not L-stable: its amplification factor tends to $-1$ for
stiff modes, so kink/discontinuity-excited high-frequency components
alternate sign and decay slowly — visible mostly in gamma near the strike.
Rannacher smoothing replaces the first time interval with two backward-Euler
half-steps (amplification $\to 0$), damping exactly those modes while
preserving global second-order convergence.

**Q6. Why bisection rather than Newton for implied vol in this project?**
Determinism and robustness across four languages: 100 halvings of a fixed
bracket $[10^{-9}, 5]$ give ~1e-10 accuracy with no dependence on vega
(which vanishes in the wings and can send Newton off the bracket). Every
port produces the same result to machine-comparable accuracy.

**Q7. How does the Brownian-bridge barrier correction work, and what bias
does it remove?**
Discrete monitoring misses barrier crossings between grid dates, biasing
knock-out prices upward with error $O(\sqrt{\Delta t})$. Conditional on step
endpoints below the barrier, the crossing probability of the bridge is
$\exp(-2(b - x_0)(b - x_1)/(\sigma^2 \Delta t))$; multiplying each path's
weight by the survival probability removes the leading-order bias with no
extra randomness, keeping the estimator deterministic per seed.

**Q8. Why does local vol misprice forward-start options?**
The forward smile generated by local vol is much flatter than today's smile:
the model's future local-vol patches are flatter than the front, and
conditioning on the future spot level averages away skew. Real markets
re-quote steep smiles every day, so cliquet-style payoffs keyed on future
smiles are underpriced. This is a *dynamics* failure — vanillas today are
still matched perfectly.

**Q9. PSOR versus penalty methods for American options — trade-offs?**
PSOR solves the discrete LCP to an explicit tolerance with no artificial
parameter, warm-starts naturally, and keeps the unmodified tridiagonal
matrix; but it is iterative per time step and slows on very fine grids.
Penalty/operator-splitting methods are one solve per step and vectorize
better, but introduce a penalty parameter that trades accuracy against
conditioning. On 200-node teaching grids PSOR's transparency wins.

**Q10. Your Dupire cap-counter reports a few hundred hits on a smooth
surface — do you ship it?**
First find *where* they fire, because "far wings, cosmetic" is a tempting
and often wrong diagnosis. In an earlier revision of this very project the
hits were assumed to come from PDE nodes 4–6 standard deviations out, but
they actually fired *at the last quoted strike* (2.5 sd from ATM), where
the flat-wing extrapolation kinks $w$ and a straddling stencil turns the
denominator negative — and they accounted for 13 of the 16 bp round-trip
error. With the stencil clamped inside the quoted box the counters read
zero on a smooth surface, so any remaining clamp means genuine
butterfly/calendar arbitrage in the input: re-fit the slice, do not ship
the number.

---

## 9. Further reading

* B. Dupire, *Pricing with a Smile*, Risk, 1994 — the original forward
  equation.
* E. Derman, I. Kani, *Riding on a Smile*, Risk, 1994 — the tree
  formulation.
* J. Gatheral, *The Volatility Surface: A Practitioner's Guide*, Wiley,
  2006 — the total-variance form used here; smiles, SVI, forward-smile
  dynamics.
* J. Gatheral, A. Jacquier, *Arbitrage-free SVI volatility surfaces*,
  Quantitative Finance, 2014 — the SSVI family generating the bundled data.
* H. Berestycki, J. Busca, I. Florent, *Asymptotics and calibration of
  local volatility models*, Quantitative Finance, 2002 — the $T \to 0$
  limit of local vol used for the $T = 0$ slice (§2.4).
* R. Rannacher, *Finite element solution of diffusion problems with
  irregular data*, Numerische Mathematik, 1984 — the smoothing start.
* M. B. Giles, R. Carter, *Convergence analysis of Crank–Nicolson and
  Rannacher time-marching*, Journal of Computational Finance, 2006 — why
  four implicit half-steps are needed for second-order Greeks (§4.3).
* D. Tavella, C. Randall, *Pricing Financial Instruments: The Finite
  Difference Method*, Wiley, 2000 — grids, boundaries, American options.
* P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer,
  2004 — Euler bias, antithetics, barrier corrections (Brownian bridge).
* P. Hagan, D. Kumar, A. Lesniewski, D. Woodward, *Managing Smile Risk*,
  Wilmott, 2002 — the classic critique of local-vol smile dynamics.
* I. Gyöngy, *Mimicking the one-dimensional marginal distributions of
  processes having an Itô differential*, PTRF, 1986 — the marginals
  theorem behind LSV calibration.
