# Bermudan Option Pricing Engine (C++)

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

## Layout

```
src/
  payoff.hpp        # option payoff (intrinsic value)         [stage 1]
  params.hpp        # MarketParams + MCConfig                 [stage 1]
  gbm.hpp           # geometric Brownian motion path gen      [stage 2]
  blackscholes.cpp  # European closed form (validation)       [stage 2]
  binomial.cpp      # Cox-Ross-Rubinstein tree (validation)   [stage 3]
  lsm.cpp           # Longstaff-Schwartz core                 [stage 4]
  greeks.cpp        # delta/gamma by finite difference        [stage 6]
  benchmark.cpp     # SE / wall-time table                    [stage 8]
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

Results (fill in from your own run — hardware dependent):

| method                | paths | price | std error | se·√N | wall (ms) |
|-----------------------|------:|------:|----------:|------:|----------:|
| plain MC              |  10^3 |       |           |       |           |
| antithetic            |  10^3 |       |           |       |           |
| antithetic + control  |  10^3 |       |           |       |           |
| plain MC              |  10^4 |       |           |       |           |
| …                     |       |       |           |       |           |
| antithetic + control  |  10^6 |       |           |       |           |

## Stages

The engine was built in the following order (each stage has its own tests):

1. CMake skeleton + payoff/parameter types
2. GBM path generator + Black-Scholes closed form
3. Cox-Ross-Rubinstein binomial tree
4. Longstaff-Schwartz core
5. Antithetic + control variate
6. Greeks (delta, gamma) by finite difference
7. Threaded, deterministic path generation
8. Benchmark harness
