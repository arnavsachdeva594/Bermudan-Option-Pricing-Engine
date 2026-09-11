#include "blackscholes.hpp"

#include <cmath>

namespace pricer {

double normal_cdf(double x) {
    // N(x) = P(Z <= x) for Z ~ N(0,1).
    // erfc(y) = 2/sqrt(pi) * integral_y^inf e^{-t^2} dt, and the change of
    // variables gives N(x) = 0.5 * erfc(-x / sqrt(2)).
    static const double INV_SQRT2 = 0.7071067811865475244;  // 1/sqrt(2)
    return 0.5 * std::erfc(-x * INV_SQRT2);
}

double black_scholes_price(const Payoff& payoff, const MarketParams& m) {
    const double S0 = m.S0;
    const double K  = payoff.strike;
    const double r  = m.r;
    const double T  = m.T;
    const double vol_sqrt_T = m.sigma * std::sqrt(T);

    const double discount = std::exp(-r * T);

    // Degenerate limit: no volatility (or no time) left. The option is worth
    // its intrinsic value on the forward, discounted back. This also keeps the
    // finite-difference Greeks well-behaved at extreme bumps.
    if (vol_sqrt_T <= 0.0) {
        const double forward = S0 * std::exp(r * T);
        return discount * payoff.intrinsic(forward);
    }

    const double d1 = (std::log(S0 / K) + (r + 0.5 * m.sigma * m.sigma) * T) / vol_sqrt_T;
    const double d2 = d1 - vol_sqrt_T;

    if (payoff.type == OptionType::Call) {
        return S0 * normal_cdf(d1) - K * discount * normal_cdf(d2);
    }
    // Put
    return K * discount * normal_cdf(-d2) - S0 * normal_cdf(-d1);
}

}  // namespace pricer
