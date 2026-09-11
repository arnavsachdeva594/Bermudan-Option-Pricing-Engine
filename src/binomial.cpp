#include "binomial.hpp"

#include <cmath>
#include <vector>

namespace pricer {

double binomial_price(const Payoff& payoff,
                      const MarketParams& m,
                      std::size_t num_steps,
                      bool american) {
    const std::size_t n = num_steps;
    const double dt = m.T / static_cast<double>(n);

    const double u = std::exp(m.sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double disc = std::exp(-m.r * dt);
    const double p = (std::exp(m.r * dt) - d) / (u - d);  // risk-neutral up-prob

    // Precompute u^k for k in [-n, n] by cumulative multiplication (fast and
    // more stable than calling pow at every node). The spot at tree node (i,j)
    // -- step i, with j up-moves -- is S0 * u^(2j - i), since d = 1/u.
    std::vector<double> upow(2 * n + 1);
    upow[n] = 1.0;
    for (std::size_t k = 1; k <= n; ++k) {
        upow[n + k] = upow[n + k - 1] * u;
        upow[n - k] = upow[n - k + 1] * d;
    }
    auto spot = [&](std::size_t i, std::size_t j) {
        // exponent 2j - i lies in [-n, n]; index into upow with the +n offset.
        const long exp = static_cast<long>(2 * j) - static_cast<long>(i);
        return m.S0 * upow[static_cast<std::size_t>(exp + static_cast<long>(n))];
    };

    // Terminal layer: option value = payoff at each terminal node.
    std::vector<double> value(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        value[j] = payoff.intrinsic(spot(n, j));
    }

    // Backward induction. After processing step i, value[0..i] holds the option
    // values at that layer; value[j] is the node with j up-moves.
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double cont = disc * (p * value[j + 1] + (1.0 - p) * value[j]);
            if (american) {
                value[j] = std::max(payoff.intrinsic(spot(i, j)), cont);
            } else {
                value[j] = cont;
            }
        }
    }
    return value[0];
}

}  // namespace pricer
