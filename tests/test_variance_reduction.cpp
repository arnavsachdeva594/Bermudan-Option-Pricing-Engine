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

namespace {
MarketParams spec_market() { return MarketParams{}; }
MCConfig vr_config(std::uint64_t seed) {
    MCConfig c;
    c.num_paths = 100'000;
    c.num_steps = 50;
    c.seed = seed;
    return c;
}
}  // namespace

// Antithetic sampling cuts the standard error versus plain MC at the same path
// count. (Relational check -- no absolute numbers.)
TEST_CASE("Antithetic reduces standard error", "[vr]") {
    MarketParams m = spec_market();
    MCConfig cfg = vr_config(101);
    Payoff put{OptionType::Put, m.K};
    PolynomialBasis basis(2);

    Estimate plain = price_bermudan(m, cfg, put, basis, {false, false});
    Estimate anti  = price_bermudan(m, cfg, put, basis, {true, false});

    REQUIRE(anti.std_error < plain.std_error);
}

// The control variate cuts the standard error further (Bermudan and European
// put payoffs are strongly correlated, so beta*correction removes most noise).
TEST_CASE("Control variate reduces standard error", "[vr]") {
    MarketParams m = spec_market();
    MCConfig cfg = vr_config(202);
    Payoff put{OptionType::Put, m.K};
    PolynomialBasis basis(2);

    Estimate plain        = price_bermudan(m, cfg, put, basis, {false, false});
    Estimate control      = price_bermudan(m, cfg, put, basis, {false, true});
    Estimate anti_control = price_bermudan(m, cfg, put, basis, {true, true});

    REQUIRE(control.std_error < plain.std_error);
    REQUIRE(anti_control.std_error < plain.std_error);
}

// Variance reduction must not bias the price: all schemes agree with plain MC
// within a few standard errors, and stay in the European..American band.
TEST_CASE("Variance reduction is unbiased", "[vr]") {
    MarketParams m = spec_market();
    MCConfig cfg = vr_config(303);
    Payoff put{OptionType::Put, m.K};
    PolynomialBasis basis(2);

    Estimate plain        = price_bermudan(m, cfg, put, basis, {false, false});
    Estimate anti_control = price_bermudan(m, cfg, put, basis, {true, true});

    const double euro = black_scholes_price(put, m);
    const double amer = binomial_price(put, m, 5000, /*american=*/true);

    // Agreement between schemes (use the larger, plain SE as the yardstick).
    REQUIRE_THAT(anti_control.price, WithinAbs(plain.price, 4.0 * plain.std_error));

    // Still a valid Bermudan price: European <= price <= American.
    REQUIRE(anti_control.price > euro - 3.0 * anti_control.std_error);
    REQUIRE(anti_control.price < amer + 3.0 * anti_control.std_error);
}

// Directly exercise the reduce() estimator: with a control variate that is a
// noisy copy of the signal (known mean), the standard error must drop.
TEST_CASE("reduce() control-variate mechanics", "[vr]") {
    // Y = base + noise; C = base (known mean 0). C explains most of Y's
    // variance, so the control-adjusted estimator has far smaller SE.
    std::vector<double> Y, C;
    std::mt19937_64 rng(7);
    std::normal_distribution<> base(0.0, 1.0), eps(0.0, 0.1);
    for (int i = 0; i < 5000; ++i) {
        const double b = base(rng);
        C.push_back(b);            // control, true mean 0
        Y.push_back(b + eps(rng)); // signal, correlated with control
    }
    Estimate no_cv = reduce(Y, Antithetic::Off);
    Estimate cv    = reduce(Y, Antithetic::Off, &C, /*control_mean=*/0.0);

    REQUIRE(cv.std_error < no_cv.std_error);
    // Both estimate the same mean (~0) closely.
    REQUIRE_THAT(cv.price, WithinAbs(no_cv.price, 4.0 * no_cv.std_error));
}
