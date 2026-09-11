#pragma once

#include "params.hpp"
#include "payoff.hpp"

namespace pricer {

// Standard normal CDF, computed from std::erfc:  N(x) = 0.5 * erfc(-x/sqrt(2)).
// Using erfc (rather than 1 - 0.5*erf) keeps the far-left tail accurate by
// avoiding the cancellation of computing 1 - (almost 1).
double normal_cdf(double x);

// Black-Scholes-Merton closed-form price of a European call or put.
// This is the exact-answer oracle used to validate the Monte-Carlo engine in
// the European case. Handles the sigma*sqrt(T) -> 0 degenerate limit by
// returning the discounted intrinsic value.
double black_scholes_price(const Payoff& payoff, const MarketParams& m);

}  // namespace pricer
