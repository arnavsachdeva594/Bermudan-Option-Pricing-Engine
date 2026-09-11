#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "binomial.hpp"
#include "blackscholes.hpp"
#include "greeks.hpp"
#include "lsm.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;
using Catch::Matchers::WithinAbs;

namespace {
double normal_pdf(double x) {
    static const double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}
// Analytic Black-Scholes Greeks, for validating the finite-difference engine.
Greeks analytic_bs_greeks(const Payoff& p, const MarketParams& m) {
    const double vst = m.sigma * std::sqrt(m.T);
    const double d1 = (std::log(m.S0 / p.strike) + (m.r + 0.5 * m.sigma * m.sigma) * m.T) / vst;
    Greeks g;
    g.delta = (p.type == OptionType::Call) ? normal_cdf(d1) : normal_cdf(d1) - 1.0;
    g.gamma = normal_pdf(d1) / (m.S0 * vst);
    return g;
}
}  // namespace

// Finite-difference Greeks on the smooth Black-Scholes price match the analytic
// Greeks (central differences are O(h^2) accurate; no MC noise here).
TEST_CASE("Finite-difference Greeks match analytic Black-Scholes", "[greeks]") {
    MarketParams m;

    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        Payoff payoff{t, m.K};
        auto price_fn = [&](const MarketParams& mm) {
            return black_scholes_price(payoff, mm);
        };
        Greeks fd = finite_difference_greeks(price_fn, m, 0.01);
        Greeks an = analytic_bs_greeks(payoff, m);

        REQUIRE_THAT(fd.delta, WithinAbs(an.delta, 1e-3));
        REQUIRE_THAT(fd.gamma, WithinAbs(an.gamma, 1e-3));
    }
}

// Sign/magnitude sanity: put delta in (-1,0), call delta in (0,1), gamma > 0.
TEST_CASE("Greek signs are correct", "[greeks]") {
    MarketParams m;

    auto put_fn  = [&](const MarketParams& mm) {
        return black_scholes_price(Payoff{OptionType::Put, m.K}, mm);
    };
    auto call_fn = [&](const MarketParams& mm) {
        return black_scholes_price(Payoff{OptionType::Call, m.K}, mm);
    };
    Greeks put  = finite_difference_greeks(put_fn, m);
    Greeks call = finite_difference_greeks(call_fn, m);

    REQUIRE(put.delta < 0.0);
    REQUIRE(put.delta > -1.0);
    REQUIRE(call.delta > 0.0);
    REQUIRE(call.delta < 1.0);
    REQUIRE(put.gamma > 0.0);
    REQUIRE(call.gamma > 0.0);
}

// LSM Bermudan-put delta, computed under common random numbers (same seed for
// every bump), is stable and close to the American tree's delta. This is the
// end-to-end test of finite-difference Greeks on the Monte-Carlo engine.
TEST_CASE("LSM finite-difference delta matches the tree (common random numbers)", "[greeks][lsm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 100'000;
    cfg.num_steps = 50;
    cfg.seed = 555;  // FIXED: shared across all three bump repricings (CRN)

    Payoff put{OptionType::Put, m.K};
    PolynomialBasis basis(2);

    // Same cfg (hence same seed) on every repricing -> common random numbers.
    auto lsm_fn = [&](const MarketParams& mm) {
        return price_bermudan(mm, cfg, put, basis, {/*antithetic=*/true, /*control=*/true}).price;
    };
    Greeks lsm = finite_difference_greeks(lsm_fn, m, 0.01);

    // Reference: American put delta from the binomial tree.
    auto tree_fn = [&](const MarketParams& mm) {
        return binomial_price(put, mm, 5000, /*american=*/true);
    };
    Greeks tree = finite_difference_greeks(tree_fn, m, 0.01);

    REQUIRE(lsm.delta < 0.0);
    REQUIRE(lsm.delta > -1.0);
    REQUIRE_THAT(lsm.delta, WithinAbs(tree.delta, 0.03));
    REQUIRE(lsm.gamma > 0.0);
}
