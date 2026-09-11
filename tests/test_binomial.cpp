#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "binomial.hpp"
#include "blackscholes.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;
using Catch::Matchers::WithinAbs;

// The stage-3 headline: the European binomial tree converges to Black-Scholes.
// CRR convergence is O(1/n) but oscillates, so we use many steps and a modest
// tolerance rather than an equality.
TEST_CASE("European binomial tree converges to Black-Scholes", "[binomial]") {
    MarketParams m;

    for (OptionType t : {OptionType::Call, OptionType::Put}) {
        Payoff payoff{t, m.K};
        const double bs = black_scholes_price(payoff, m);
        const double tree = binomial_price(payoff, m, 5000, /*american=*/false);
        REQUIRE_THAT(tree, WithinAbs(bs, 5e-3));
    }
}

// Finer trees are more accurate: the 5000-step error should be much smaller
// than the 50-step error (checks convergence direction without magic numbers).
TEST_CASE("Binomial accuracy improves with more steps", "[binomial]") {
    MarketParams m;
    Payoff put{OptionType::Put, m.K};
    const double bs = black_scholes_price(put, m);

    const double err_coarse = std::abs(binomial_price(put, m, 50, false) - bs);
    const double err_fine   = std::abs(binomial_price(put, m, 5000, false) - bs);
    REQUIRE(err_fine < err_coarse);
}

// No-dividend American call is never exercised early, so it equals the
// European call (and hence Black-Scholes). Identity -> no hardcoded number.
TEST_CASE("American call equals European call with no dividends", "[binomial]") {
    MarketParams m;
    Payoff call{OptionType::Call, m.K};

    const double amer = binomial_price(call, m, 2000, /*american=*/true);
    const double euro = binomial_price(call, m, 2000, /*american=*/false);
    REQUIRE_THAT(amer, WithinAbs(euro, 1e-10));
}

// The American put carries a strictly positive early-exercise premium.
TEST_CASE("American put exceeds European put", "[binomial]") {
    MarketParams m;
    Payoff put{OptionType::Put, m.K};

    const double amer = binomial_price(put, m, 2000, /*american=*/true);
    const double euro = binomial_price(put, m, 2000, /*american=*/false);
    REQUIRE(amer > euro);
    // And the European tree still matches Black-Scholes.
    REQUIRE_THAT(euro, WithinAbs(black_scholes_price(put, m), 5e-3));
}
