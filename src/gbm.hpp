#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "params.hpp"

namespace pricer {

// Full simulated stock paths under risk-neutral Geometric Brownian Motion.
//
// S is (num_paths) x (num_steps + 1): row i is one path, column 0 is S0, and
// column j is the spot at time t_j = j*dt. We store the FULL path (not just the
// terminal value) because Longstaff-Schwartz sweeps backward through the
// exercise dates and needs the spot at every date on every path.
struct Paths {
    Eigen::MatrixXd S;      // (num_paths) x (num_steps + 1)
    double          dt = 0.0;

    [[nodiscard]] Eigen::Index num_paths() const { return S.rows(); }
    [[nodiscard]] Eigen::Index num_steps() const { return S.cols() - 1; }

    [[nodiscard]] auto spot_at(Eigen::Index j) const { return S.col(j); }
    [[nodiscard]] auto terminal() const { return S.col(S.cols() - 1); }
};

namespace detail {

// splitmix64: turns a counter into a well-spread 64-bit value, so that seeding
// per-path RNGs from sequential indices does not produce correlated streams.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Deterministic seed for work-unit `index` under a given master seed. Depends
// only on (master_seed, index) -- never on thread count or partitioning -- so
// the generated paths are identical regardless of how work is split.
inline std::uint64_t unit_seed(std::uint64_t master, std::size_t index) {
    return splitmix64(master + splitmix64(static_cast<std::uint64_t>(index)));
}

inline unsigned resolve_threads(unsigned requested) {
    if (requested != 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : hw;
}

// Run f(begin, end) over a partition of [0, total) into contiguous chunks on
// `nthreads` threads. Because each work-unit is seeded independently, the chunk
// boundaries do not affect any unit's output.
template <class F>
void parallel_for(std::size_t total, unsigned nthreads, F&& f) {
    nthreads = std::max(1u, nthreads);
    if (nthreads == 1 || total == 0) {
        f(std::size_t{0}, total);
        return;
    }
    nthreads = static_cast<unsigned>(std::min<std::size_t>(nthreads, total));
    const std::size_t chunk = (total + nthreads - 1) / nthreads;

    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (unsigned t = 1; t < nthreads; ++t) {
        const std::size_t b = t * chunk;
        const std::size_t e = std::min(total, b + chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    f(std::size_t{0}, std::min(total, chunk));  // this thread does chunk 0
    for (auto& th : pool) th.join();
}

}  // namespace detail

// Generate GBM paths under the risk-neutral measure (drift = r), in parallel.
//
// Exact stepping (NOT Euler): each step applies the closed-form GBM solution
//     S(t+dt) = S(t) * exp( (r - sigma^2/2) dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)
// so there is no time-discretization error in the stock process itself.
//
// Each PATH is seeded independently from (cfg.seed, path index), so the output
// is bit-for-bit identical for any cfg.num_threads -- reproducibility does not
// depend on the core count. (Cost: one mt19937_64 seeded per path; a
// counter-based RNG such as Philox would remove that overhead.)
inline Paths generate_gbm_paths(const MarketParams& m, const MCConfig& cfg) {
    const Eigen::Index N = static_cast<Eigen::Index>(cfg.num_paths);
    const Eigen::Index M = static_cast<Eigen::Index>(cfg.num_steps);

    Paths paths;
    paths.dt = m.T / static_cast<double>(M);
    paths.S.resize(N, M + 1);
    paths.S.col(0).setConstant(m.S0);

    const double drift = (m.r - 0.5 * m.sigma * m.sigma) * paths.dt;
    const double vol   = m.sigma * std::sqrt(paths.dt);

    detail::parallel_for(
        static_cast<std::size_t>(N), detail::resolve_threads(cfg.num_threads),
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                std::mt19937_64            rng(detail::unit_seed(cfg.seed, i));
                std::normal_distribution<> gauss(0.0, 1.0);
                double s = m.S0;
                for (Eigen::Index j = 1; j <= M; ++j) {
                    s *= std::exp(drift + vol * gauss(rng));
                    paths.S(static_cast<Eigen::Index>(i), j) = s;
                }
            }
        });
    return paths;
}

// Antithetic path generation (parallel). Produces cfg.num_paths paths arranged
// in mirror pairs: within pair p, the shocks Z drive row 2p and -Z drive row
// 2p+1. Because a put payoff is monotone in the shocks, the two paths' payoffs
// are negatively correlated, which the pair-aware estimator exploits.
//
// The work-unit is the PAIR (so both halves share the same Z draws), seeded
// from (cfg.seed, pair index) -- again independent of thread count.
// cfg.num_paths must be even; the last path is dropped if it is odd.
inline Paths generate_gbm_paths_antithetic(const MarketParams& m, const MCConfig& cfg) {
    const Eigen::Index P = static_cast<Eigen::Index>(cfg.num_paths / 2);  // pairs
    const Eigen::Index N = 2 * P;
    const Eigen::Index M = static_cast<Eigen::Index>(cfg.num_steps);

    Paths paths;
    paths.dt = m.T / static_cast<double>(M);
    paths.S.resize(N, M + 1);
    paths.S.col(0).setConstant(m.S0);

    const double drift = (m.r - 0.5 * m.sigma * m.sigma) * paths.dt;
    const double vol   = m.sigma * std::sqrt(paths.dt);

    detail::parallel_for(
        static_cast<std::size_t>(P), detail::resolve_threads(cfg.num_threads),
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t p = begin; p < end; ++p) {
                std::mt19937_64            rng(detail::unit_seed(cfg.seed, p));
                std::normal_distribution<> gauss(0.0, 1.0);
                double s_plus = m.S0, s_minus = m.S0;
                const Eigen::Index row = static_cast<Eigen::Index>(2 * p);
                for (Eigen::Index j = 1; j <= M; ++j) {
                    const double z = gauss(rng);
                    s_plus  *= std::exp(drift + vol * z);
                    s_minus *= std::exp(drift - vol * z);
                    paths.S(row,     j) = s_plus;
                    paths.S(row + 1, j) = s_minus;
                }
            }
        });
    return paths;
}

}  // namespace pricer
