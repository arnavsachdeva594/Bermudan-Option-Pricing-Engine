#pragma once

#include <cstddef>

#include "params.hpp"
#include "payoff.hpp"

namespace pricer {

// Cox-Ross-Rubinstein binomial-tree price.
//
// Discretizes [0,T] into num_steps steps and prices by backward induction on a
// recombining lattice. With american=false this converges to Black-Scholes as
// num_steps grows (our European cross-check); with american=true it is the
// essentially-exact American oracle that upper-bounds the Bermudan LSM price.
//
// CRR parameters:
//   u = e^{sigma*sqrt(dt)},  d = 1/u,  p = (e^{r*dt} - d)/(u - d)
// chosen so the lattice matches GBM's per-step mean and variance under the
// risk-neutral measure.
double binomial_price(const Payoff& payoff,
                      const MarketParams& m,
                      std::size_t num_steps,
                      bool american);

}  // namespace pricer
