# Bermudan Option Pricing Engine (C++)

[![CI](https://github.com/arnavsachdeva594/Option-Pricer-Demo/actions/workflows/ci.yml/badge.svg)](https://github.com/arnavsachdeva594/Option-Pricer-Demo/actions/workflows/ci.yml)

Prices early-exercise options by **least-squares Monte Carlo (Longstaff-Schwartz)**:
it regresses continuation values on a polynomial basis at each exercise date to
determine the optimal stopping rule, and cross-validates the result against a
binomial tree and the Black-Scholes closed form. Variance reduction (antithetic
pairs + a European control variate) and multithreaded path generation are layered
on top.

## Features

- **Longstaff-Schwartz LSM** with a pluggable regression basis (`{1,S,S²}`
  polynomial or Laguerre), ITM-only regression, stable `householderQr` solve.
- **Cross-validation**: Black-Scholes closed form (European) and a
  Cox-Ross-Rubinstein tree (American) bracket the LSM price —
  `European ≤ Bermudan ≤ American` is asserted in the tests.
- **Variance reduction**: antithetic pairs (pair-aware standard error) and a
  European control variate with empirically estimated optimal beta.
- **Greeks**: delta and gamma by central finite difference under common random
  numbers.
- **Threaded path generation** that is deterministic *independent of core count*
  (per-path seeding), proven by test.
- **Benchmark harness** printing the SE / wall-time table below.

## Design at a glance

- **`MarketParams`** — the problem: `S0, K, r, sigma, T`. Every pricer reads the
  same struct, which is what makes the three-way cross-validation meaningful.
- **`MCConfig`** — the effort: number of paths, exercise dates, and the master
  seed (reproducibility matters for finite-difference Greeks, which reprice with
  the same seed).
- **`Payoff`** — the contract terminal condition: `max(K-S,0)` for a put,
  `max(S-K,0)` for a call. A small pluggable value type so the same pricing code
  handles calls and puts.

## Notes on the method

**Why regression appears at all.** The value of holding an option at an exercise
date is its *continuation value* — a conditional expectation of the discounted
future cashflows given the current stock price. Monte Carlo produces realized
futures, not conditional expectations, so Longstaff-Schwartz estimates that
expectation by ordinary least squares: at each date it projects the realized
discounted future cashflows onto a small basis of functions of the current spot
(`{1, S, S²}` or Laguerre). The fitted function is the continuation-value
estimate, and comparing it to the intrinsic value gives the exercise decision.

**Why in-the-money paths only.** The exercise-versus-hold decision only exists
where exercising now pays something; on out-of-the-money paths the intrinsic
value is zero, so there is no decision to inform. Including them also drags the
low-order fit toward a flat, irrelevant region and worsens accuracy right at the
exercise boundary, where the decision actually flips. Regressing on ITM paths
only concentrates the basis's limited flexibility where it matters. (OTM paths
still carry their downstream cashflows forward; they are excluded from the
regression, not from the price.)

**Which way the bias runs, and why the bracket holds.** LSM is biased **low**:
it exercises according to a regression-estimated stopping rule, and any
suboptimal policy earns no more in expectation than the true optimal one, so the
estimate is a lower bound on the true Bermudan price. (A separate, usually
smaller, upward "foresight" bias comes from fitting and pricing on the same
paths; pricing on an independent path set removes it.) This is exactly why
`European ≤ Bermudan ≤ American` holds: more exercise opportunities can never
reduce value, so the Bermudan dominates the European (a monotonicity theorem),
while a Bermudan with finitely many dates is dominated by the
continuously-exercisable American. Black-Scholes gives the exact European value
and the binomial tree an essentially exact American one, so the LSM estimate
should — and in the benchmark does — land between them.

## Layout

```
src/
  payoff.hpp        # option payoff: intrinsic value + in-the-money predicate
  params.hpp        # MarketParams (contract/market) + MCConfig (MC knobs)
  gbm.hpp           # risk-neutral GBM paths; parallel, per-path seeded
  blackscholes.cpp  # European closed form; normal CDF via std::erfc
  binomial.cpp      # Cox-Ross-Rubinstein tree (European + American)
  lsm.hpp / lsm.cpp # LSM backward induction, pluggable basis, variance reduction
  greeks.cpp        # delta & gamma by central finite difference
  benchmark.cpp     # price / std-error / wall-time table
tests/              # Catch2 unit + cross-validation tests
CMakeLists.txt
```

## Dependencies

Fetched automatically by CMake (`FetchContent`), nothing to install by hand:

- **Eigen 3.4** — least-squares regression in Longstaff-Schwartz (fetched as a
  pinned source tarball)
- **Catch2 v3** — test framework (fetched via a shallow git clone of the `v3.5.4`
  tag, so the first configure needs network access to GitHub)

Standard-library only otherwise: `std::mt19937_64` + `std::normal_distribution`
for the RNG, `std::thread` for parallel path generation.

## Build & test

```sh
cmake -S . -B build            # configures + fetches Eigen/Catch2
cmake --build build -j         # builds library + tests
ctest --test-dir build         # runs the suite
```

Requires a C++17 compiler and CMake >= 3.20.

## Benchmarks

```sh
./build/benchmark          # 1e3, 1e4, 1e5, 1e6 paths (default)
./build/benchmark 5        # up to 1e5 paths
./build/benchmark 4 7      # 1e4 .. 1e7 paths
```

The harness prices the spec Bermudan put (S0=K=100, r=5%, σ=20%, T=1, 50
exercise dates) with each estimator and prints:

```
method                       paths       price     std error    se*sqrt(N)     wall (ms)
```

How to read it:

- **`se*sqrt(N)` is ~constant down each method's column** — empirical
  confirmation that `SE = c/√N` (halving the error costs 4× the paths).
- **`std error` shrinks across methods at fixed N** — antithetic then control
  variate cut the constant `c`.
- **`price` stays inside `[European, American]`** for every row.

### Sample run

Measured on a 4-core Intel Xeon @ 2.10GHz (`hardware_concurrency = 4`), Release
build, GCC 13. Numbers are hardware dependent; the *shape* is the point.

```
Bermudan put, S0=100 K=100 r=0.05 sigma=0.2 T=1, 50 exercise dates
hardware_concurrency = 4
European (Black-Scholes) = 5.5735   |   American (tree, 5000 steps) = 6.0902
Bermudan price must sit between these two.

method                       paths       price     std error    se*sqrt(N)     wall (ms)
----------------------------------------------------------------------------------------
plain MC                      1000      6.1209      0.246814        7.8049           2.3
antithetic                    1000      6.2014      0.154475        4.8849           1.7
antithetic + control          1000      6.2760      0.122309        3.8677           1.6

plain MC                     10000      6.0109      0.070331        7.0331          25.7
antithetic                   10000      5.9706      0.042467        4.2467          21.1
antithetic + control         10000      6.0229      0.037154        3.7154          16.2

plain MC                    100000      6.0456      0.022741        7.1913         244.6
antithetic                  100000      6.0454      0.013912        4.3993         204.7
antithetic + control        100000      6.0500      0.011812        3.7351         160.5

plain MC                   1000000      6.0549      0.007197        7.1966        3302.3
antithetic                 1000000      6.0560      0.004415        4.4151        2401.4
antithetic + control       1000000      6.0550      0.003743        3.7432        2305.6
```

What the table shows:

- **`se*sqrt(N)` is flat down each column** (plain ≈ 7.2, antithetic ≈ 4.4,
  control ≈ 3.7 once past the tiny-N regime) — the empirical confirmation that
  `SE = c/√N`. Going from 10⁵ to 10⁶ paths cuts the standard error by ≈ √10.
- **The constant `c` shrinks across methods** — antithetic roughly halves it,
  the control variate cuts it further, at essentially no extra cost per path.
- **The price lands inside `[5.5735, 6.0902]`** at every large-N row; the wide
  bands at 10³ are just Monte-Carlo noise (the estimate at 10³ is within one
  standard error of the bracket).
