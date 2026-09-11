#pragma once

#include <cstddef>
#include <cstdint>

namespace pricer {

// MarketParams describes the *problem*: the contract and the market it lives
// in. These five numbers fully determine the Black-Scholes world, so every
// pricer in this project (closed-form BS, binomial tree, Longstaff-Schwartz
// Monte Carlo) reads the same struct. That shared input is what makes the
// three-way cross-validation meaningful -- they price the *same* contract.
//
// Note: the strike K also lives on Payoff. K here is the canonical contract
// strike; construct a Payoff as {type, params.K} when a pricer needs one.
struct MarketParams {
    double S0    = 100.0;  // spot price today
    double K     = 100.0;  // strike price
    double r     = 0.05;   // risk-free rate (continuously compounded), for discounting
    double sigma = 0.20;   // volatility (annualized)
    double T     = 1.0;    // time to maturity in years
};

// MCConfig describes *how hard we try* to solve the problem: the Monte-Carlo
// discretization knobs. These are not real-world quantities -- they trade
// compute for accuracy. Kept separate from MarketParams so the contract can be
// held fixed while sweeping simulation effort (see the benchmark harness).
struct MCConfig {
    std::size_t   num_paths = 100'000;  // number of simulated GBM paths
    std::size_t   num_steps = 50;       // time steps == Bermudan exercise dates
    std::uint64_t seed      = 12345ULL; // master seed: makes every run reproducible
};

}  // namespace pricer
