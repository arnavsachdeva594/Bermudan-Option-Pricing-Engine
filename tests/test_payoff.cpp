#include <catch2/catch_test_macros.hpp>

#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;

TEST_CASE("Put intrinsic value", "[payoff]") {
    Payoff put{OptionType::Put, 100.0};

    SECTION("in the money when spot below strike") {
        REQUIRE(put.intrinsic(80.0) == 20.0);
        REQUIRE(put.in_the_money(80.0));
    }
    SECTION("worthless to exercise when spot at or above strike") {
        REQUIRE(put.intrinsic(100.0) == 0.0);
        REQUIRE(put.intrinsic(120.0) == 0.0);
        REQUIRE_FALSE(put.in_the_money(100.0));  // ATM is not strictly ITM
        REQUIRE_FALSE(put.in_the_money(120.0));
    }
}

TEST_CASE("Call intrinsic value", "[payoff]") {
    Payoff call{OptionType::Call, 100.0};

    REQUIRE(call.intrinsic(120.0) == 20.0);
    REQUIRE(call.in_the_money(120.0));
    REQUIRE(call.intrinsic(100.0) == 0.0);
    REQUIRE(call.intrinsic(80.0) == 0.0);
    REQUIRE_FALSE(call.in_the_money(80.0));
}

TEST_CASE("Default MarketParams match the spec", "[params]") {
    MarketParams p;
    REQUIRE(p.S0 == 100.0);
    REQUIRE(p.K == 100.0);
    REQUIRE(p.r == 0.05);
    REQUIRE(p.sigma == 0.20);
    REQUIRE(p.T == 1.0);
}

TEST_CASE("Default MCConfig matches the spec", "[params]") {
    MCConfig c;
    REQUIRE(c.num_paths == 100'000);
    REQUIRE(c.num_steps == 50);
}
