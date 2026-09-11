#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <vector>

#include "gbm.hpp"
#include "lsm.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;

namespace {
MCConfig cfg_with_threads(unsigned nthreads) {
    MCConfig c;
    c.num_paths = 40'000;
    c.num_steps = 50;
    c.seed = 314159;
    c.num_threads = nthreads;
    return c;
}
}  // namespace

// The whole point of stage 7: the generated paths are bit-for-bit identical
// regardless of the thread count, because each path is seeded independently
// from (master seed, path index) -- not from the thread.
TEST_CASE("Plain path generation is thread-count independent", "[threading]") {
    MarketParams m;
    Paths ref = generate_gbm_paths(m, cfg_with_threads(1));

    for (unsigned nt : {2u, 3u, 4u, 8u}) {
        Paths other = generate_gbm_paths(m, cfg_with_threads(nt));
        REQUIRE(other.S.rows() == ref.S.rows());
        REQUIRE(other.S.cols() == ref.S.cols());
        // Exact bitwise equality (not "within tolerance").
        REQUIRE(other.S == ref.S);
    }
}

TEST_CASE("Antithetic path generation is thread-count independent", "[threading]") {
    MarketParams m;
    Paths ref = generate_gbm_paths_antithetic(m, cfg_with_threads(1));

    for (unsigned nt : {2u, 4u, 8u}) {
        Paths other = generate_gbm_paths_antithetic(m, cfg_with_threads(nt));
        REQUIRE(other.S == ref.S);
    }
}

// The downstream price inherits the determinism: identical paths -> identical
// LSM price, whatever the core count.
TEST_CASE("LSM price is thread-count independent", "[threading][lsm]") {
    MarketParams m;
    Payoff put{OptionType::Put, m.K};
    PolynomialBasis basis(2);

    LSMResult ref = longstaff_schwartz(generate_gbm_paths(m, cfg_with_threads(1)),
                                       put, m, basis);
    for (unsigned nt : {2u, 4u, 8u}) {
        LSMResult r = longstaff_schwartz(generate_gbm_paths(m, cfg_with_threads(nt)),
                                         put, m, basis);
        REQUIRE(r.price == ref.price);        // exact equality
        REQUIRE(r.std_error == ref.std_error);
    }
}

// Auto thread count (0 -> hardware_concurrency) also matches single-threaded.
TEST_CASE("Auto thread count matches single-threaded", "[threading]") {
    MarketParams m;
    Paths ref  = generate_gbm_paths(m, cfg_with_threads(1));
    Paths autp = generate_gbm_paths(m, cfg_with_threads(0));  // 0 => auto
    REQUIRE(autp.S == ref.S);
}
