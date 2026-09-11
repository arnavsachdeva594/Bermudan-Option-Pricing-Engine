# Bermudan Option Pricing Engine (C++)

Prices early-exercise options by **least-squares Monte Carlo (Longstaff-Schwartz)**:
it regresses continuation values on a polynomial basis at each exercise date to
determine the optimal stopping rule, and cross-validates the result against a
binomial tree and the Black-Scholes closed form. Variance reduction (antithetic
pairs + a European control variate) and multithreaded path generation are layered
on top.

> Status: built in stages. This README grows as stages land.

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
