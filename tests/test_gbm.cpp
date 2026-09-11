#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "blackscholes.hpp"
#include "gbm.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;
using Catch::Matchers::WithinAbs;

TEST_CASE("GBM path structure", "[gbm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 1000;
    cfg.num_steps = 10;

    Paths paths = generate_gbm_paths(m, cfg);

    REQUIRE(paths.num_paths() == 1000);
    REQUIRE(paths.num_steps() == 10);
    REQUIRE_THAT(paths.dt, WithinAbs(m.T / 10.0, 1e-15));

    // Every path starts at S0 and stays strictly positive (log-normal).
    for (Eigen::Index i = 0; i < paths.num_paths(); ++i) {
        REQUIRE(paths.S(i, 0) == m.S0);
        for (Eigen::Index j = 0; j <= paths.num_steps(); ++j) {
            REQUIRE(paths.S(i, j) > 0.0);
        }
    }
}

// Martingale / drift-correction test: under the risk-neutral measure the
// discounted stock is a martingale, so e^{-rT} * E[S_T] = S0. This is the
// test that fails loudly if the -sigma^2/2 Ito term is dropped (the stock
// would then drift too high).
TEST_CASE("Discounted terminal spot is a martingale", "[gbm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 200'000;
    cfg.num_steps = 50;
    cfg.seed = 2024;

    Paths paths = generate_gbm_paths(m, cfg);

    const double mean_ST = paths.terminal().mean();
    const double discounted = std::exp(-m.r * m.T) * mean_ST;

    // Standard error of the mean of S_T; allow a comfortable 5-sigma band so
    // the (fixed-seed, deterministic) test is not flaky.
    const auto col = paths.terminal();
    const double var = (col.array() - mean_ST).square().sum()
                       / static_cast<double>(col.size() - 1);
    const double se = std::sqrt(var / static_cast<double>(col.size()));

    REQUIRE_THAT(discounted, WithinAbs(m.S0, 5.0 * se));
}

// The headline cross-check for stage 2: pricing the European option by plain
// Monte Carlo (discounted mean terminal payoff) must converge to the
// Black-Scholes closed form.
TEST_CASE("European Monte Carlo converges to Black-Scholes", "[gbm][blackscholes]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 400'000;
    cfg.num_steps = 50;
    cfg.seed = 7;

    Paths paths = generate_gbm_paths(m, cfg);

    auto price_european_mc = [&](const Payoff& payoff) {
        const double disc = std::exp(-m.r * m.T);
        const Eigen::Index n = paths.num_paths();
        double sum = 0.0, sumsq = 0.0;
        for (Eigen::Index i = 0; i < n; ++i) {
            const double pv = disc * payoff.intrinsic(paths.terminal()(i));
            sum += pv;
            sumsq += pv * pv;
        }
        const double mean = sum / static_cast<double>(n);
        const double var = (sumsq / static_cast<double>(n) - mean * mean)
                           * static_cast<double>(n) / static_cast<double>(n - 1);
        const double se = std::sqrt(var / static_cast<double>(n));
        return std::pair<double, double>{mean, se};
    };

    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        Payoff payoff{t, m.K};
        auto [mc_price, se] = price_european_mc(payoff);
        const double bs = black_scholes_price(payoff, m);
        // Within 4 standard errors of the exact price.
        REQUIRE_THAT(mc_price, WithinAbs(bs, 4.0 * se));
    }
}
