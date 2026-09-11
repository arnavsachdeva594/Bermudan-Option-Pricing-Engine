#include "greeks.hpp"

namespace pricer {

Greeks finite_difference_greeks(
    const std::function<double(const MarketParams&)>& price_fn,
    const MarketParams& m,
    double h_rel) {
    const double h = h_rel * m.S0;

    MarketParams up = m;
    up.S0 = m.S0 + h;
    MarketParams down = m;
    down.S0 = m.S0 - h;

    // Three repricings. When price_fn is a Monte-Carlo pricer sharing one seed,
    // these run on the same random draws (common random numbers), so the noise
    // cancels in the differences below.
    const double v_up   = price_fn(up);
    const double v_mid  = price_fn(m);
    const double v_down = price_fn(down);

    Greeks g;
    g.delta = (v_up - v_down) / (2.0 * h);
    g.gamma = (v_up - 2.0 * v_mid + v_down) / (h * h);
    return g;
}

}  // namespace pricer
