#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "blackscholes.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("normal_cdf basic values", "[blackscholes]") {
    REQUIRE_THAT(normal_cdf(0.0), WithinAbs(0.5, 1e-12));
    // Symmetry: N(x) + N(-x) = 1.
    REQUIRE_THAT(normal_cdf(1.3) + normal_cdf(-1.3), WithinAbs(1.0, 1e-12));
    // Monotonic and bounded.
    REQUIRE(normal_cdf(-5.0) > 0.0);
    REQUIRE(normal_cdf(-5.0) < normal_cdf(0.0));
    REQUIRE(normal_cdf(5.0) < 1.0);
}

// Put-call parity is an identity of the closed form: C - P = S0 - K*e^{-rT}.
// It requires no hardcoded "known price", yet catches sign, discounting, and
// d1/d2 bugs immediately.
TEST_CASE("Black-Scholes put-call parity", "[blackscholes]") {
    MarketParams m;  // spec defaults
    Payoff call{OptionType::Call, m.K};
    Payoff put{OptionType::Put, m.K};

    const double c = black_scholes_price(call, m);
    const double p = black_scholes_price(put, m);
    const double parity = m.S0 - m.K * std::exp(-m.r * m.T);

    REQUIRE_THAT(c - p, WithinAbs(parity, 1e-10));
}

TEST_CASE("Black-Scholes sanity", "[blackscholes]") {
    MarketParams m;
    Payoff call{OptionType::Call, m.K};
    Payoff put{OptionType::Put, m.K};

    // Both prices are strictly positive for a live ATM option.
    REQUIRE(black_scholes_price(call, m) > 0.0);
    REQUIRE(black_scholes_price(put, m) > 0.0);

    // No-arbitrage lower bound for a European put: P >= K*e^{-rT} - S0.
    // A deep in-the-money put (huge strike) sits essentially on this bound
    // because its remaining time value is negligible.
    Payoff deep_put{OptionType::Put, 1000.0};
    const double price = black_scholes_price(deep_put, m);
    const double lower_bound = 1000.0 * std::exp(-m.r * m.T) - m.S0;
    REQUIRE(price >= lower_bound - 1e-9);
}

TEST_CASE("Degenerate zero-vol limit is discounted intrinsic", "[blackscholes]") {
    MarketParams m;
    m.sigma = 0.0;
    Payoff call{OptionType::Call, m.K};
    // With sigma=0 the stock grows deterministically to the forward S0*e^{rT}.
    const double forward = m.S0 * std::exp(m.r * m.T);
    const double expected = std::exp(-m.r * m.T) * std::max(forward - m.K, 0.0);
    REQUIRE_THAT(black_scholes_price(call, m), WithinAbs(expected, 1e-12));
}
