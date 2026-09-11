#pragma once

#include <functional>

#include "params.hpp"

namespace pricer {

struct Greeks {
    double delta = 0.0;  // dV/dS
    double gamma = 0.0;  // d^2V/dS^2
};

// Central finite-difference delta and gamma with respect to spot, using a
// relative bump h = h_rel * S0 (spec default h_rel = 0.01):
//
//   delta = ( V(S0+h) - V(S0-h) ) / (2h)          -- O(h^2) accurate
//   gamma = ( V(S0+h) - 2 V(S0) + V(S0-h) ) / h^2
//
// price_fn reprices the option given a (bumped) MarketParams. For a Monte-Carlo
// pricer it MUST use common random numbers (the same seed for every bump);
// otherwise independent MC noise, amplified by the 1/(2h) and 1/h^2 factors,
// swamps the derivative. Feeding price_bermudan the same MCConfig achieves this
// automatically, since paths are regenerated from the fixed seed.
Greeks finite_difference_greeks(
    const std::function<double(const MarketParams&)>& price_fn,
    const MarketParams& m,
    double h_rel = 0.01);

}  // namespace pricer
