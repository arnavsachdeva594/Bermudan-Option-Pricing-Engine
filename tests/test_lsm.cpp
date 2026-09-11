#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "binomial.hpp"
#include "blackscholes.hpp"
#include "gbm.hpp"
#include "lsm.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;
using Catch::Matchers::WithinAbs;

// THE stage-4 validation: the LSM Bermudan put must sit between the European
// price (Black-Scholes) and the American price (binomial tree):
//     European <= Bermudan(LSM) <= American
// More exercise dates cannot reduce value (>= European); LSM uses a
// suboptimal regression policy so it is a lower bound on the true Bermudan,
// which is itself <= American. Bands are a few standard errors for MC noise.
TEST_CASE("LSM Bermudan put sits between European and American", "[lsm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 100'000;
    cfg.num_steps = 50;
    cfg.seed = 42;

    Payoff put{OptionType::Put, m.K};
    Paths paths = generate_gbm_paths(m, cfg);
    PolynomialBasis basis(2);  // {1, S, S^2}
    LSMResult lsm = longstaff_schwartz(paths, put, m, basis);

    const double euro = black_scholes_price(put, m);
    const double amer = binomial_price(put, m, 5000, /*american=*/true);

    // Ordering with a small MC tolerance.
    REQUIRE(lsm.price > euro - 3.0 * lsm.std_error);
    REQUIRE(lsm.price < amer + 3.0 * lsm.std_error);
    // The early-exercise value is real for this put: LSM should be clearly
    // above the European price (well beyond noise).
    REQUIRE(lsm.price > euro);
}

// A no-dividend Bermudan CALL is never exercised early, so LSM should recover
// the European call (Black-Scholes) within Monte-Carlo error and must not
// exceed the American tree call.
TEST_CASE("LSM Bermudan call recovers the European call", "[lsm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 100'000;
    cfg.num_steps = 50;
    cfg.seed = 123;

    Payoff call{OptionType::Call, m.K};
    Paths paths = generate_gbm_paths(m, cfg);
    PolynomialBasis basis(2);
    LSMResult lsm = longstaff_schwartz(paths, call, m, basis);

    const double bs = black_scholes_price(call, m);
    REQUIRE_THAT(lsm.price, WithinAbs(bs, 4.0 * lsm.std_error));
}

// Pluggable basis: the polynomial and Laguerre bases should agree on the
// Bermudan put price to within Monte-Carlo error (same paths, different basis).
TEST_CASE("Polynomial and Laguerre bases agree", "[lsm][basis]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 100'000;
    cfg.num_steps = 50;
    cfg.seed = 2718;

    Payoff put{OptionType::Put, m.K};
    Paths paths = generate_gbm_paths(m, cfg);

    PolynomialBasis poly(2);
    LaguerreBasis   lag(m.K, 3);

    LSMResult a = longstaff_schwartz(paths, put, m, poly);
    LSMResult b = longstaff_schwartz(paths, put, m, lag);

    const double band = 4.0 * std::sqrt(a.std_error * a.std_error +
                                        b.std_error * b.std_error);
    REQUIRE_THAT(a.price, WithinAbs(b.price, band));
}

// Determinism: same paths + same basis => identical price.
TEST_CASE("LSM is deterministic for fixed paths", "[lsm]") {
    MarketParams m;
    MCConfig cfg;
    cfg.num_paths = 20'000;
    cfg.num_steps = 50;
    cfg.seed = 9;

    Payoff put{OptionType::Put, m.K};
    Paths paths = generate_gbm_paths(m, cfg);
    PolynomialBasis basis(2);

    LSMResult r1 = longstaff_schwartz(paths, put, m, basis);
    LSMResult r2 = longstaff_schwartz(paths, put, m, basis);
    REQUIRE(r1.price == r2.price);
}
