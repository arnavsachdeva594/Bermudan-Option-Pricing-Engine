#pragma once

#include <cmath>
#include <cstddef>
#include <random>

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

    // The spot values at exercise date j across all paths (a column view).
    [[nodiscard]] auto spot_at(Eigen::Index j) const { return S.col(j); }
    // Terminal spot S(T) across all paths.
    [[nodiscard]] auto terminal() const { return S.col(S.cols() - 1); }
};

// Generate GBM paths under the risk-neutral measure (drift = r).
//
// Exact stepping (NOT Euler): each step applies the closed-form GBM solution
//     S(t+dt) = S(t) * exp( (r - sigma^2/2) dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)
// so there is no time-discretization error in the stock process itself; the
// only discretization is num_steps (the number of Bermudan exercise dates).
//
// Deterministic given cfg.seed: one std::mt19937_64 drives the whole run
// (multithreading with per-thread seeded RNGs comes in a later stage).
inline Paths generate_gbm_paths(const MarketParams& m, const MCConfig& cfg) {
    const Eigen::Index N = static_cast<Eigen::Index>(cfg.num_paths);
    const Eigen::Index M = static_cast<Eigen::Index>(cfg.num_steps);

    Paths paths;
    paths.dt = m.T / static_cast<double>(M);
    paths.S.resize(N, M + 1);

    // Per-step constants: log-drift includes the -sigma^2/2 Ito correction so
    // that the PRICE (not the log-price) grows in expectation at rate r.
    const double drift = (m.r - 0.5 * m.sigma * m.sigma) * paths.dt;
    const double vol   = m.sigma * std::sqrt(paths.dt);

    std::mt19937_64            rng(cfg.seed);
    std::normal_distribution<> gauss(0.0, 1.0);

    paths.S.col(0).setConstant(m.S0);
    for (Eigen::Index i = 0; i < N; ++i) {
        double s = m.S0;
        for (Eigen::Index j = 1; j <= M; ++j) {
            const double z = gauss(rng);
            s *= std::exp(drift + vol * z);
            paths.S(i, j) = s;
        }
    }
    return paths;
}

// Antithetic path generation. Produces cfg.num_paths paths arranged in mirror
// pairs: for each pair the shocks Z drive row 2p and -Z drive row 2p+1. Because
// a put payoff is monotone in the shocks, the two paths' payoffs are negatively
// correlated, which the pair-aware estimator exploits to cut variance.
//
// cfg.num_paths must be even; the last path is dropped if it is odd.
inline Paths generate_gbm_paths_antithetic(const MarketParams& m, const MCConfig& cfg) {
    const Eigen::Index P = static_cast<Eigen::Index>(cfg.num_paths / 2);  // pairs
    const Eigen::Index N = 2 * P;
    const Eigen::Index M = static_cast<Eigen::Index>(cfg.num_steps);

    Paths paths;
    paths.dt = m.T / static_cast<double>(M);
    paths.S.resize(N, M + 1);

    const double drift = (m.r - 0.5 * m.sigma * m.sigma) * paths.dt;
    const double vol   = m.sigma * std::sqrt(paths.dt);

    std::mt19937_64            rng(cfg.seed);
    std::normal_distribution<> gauss(0.0, 1.0);

    paths.S.col(0).setConstant(m.S0);
    for (Eigen::Index p = 0; p < P; ++p) {
        double s_plus = m.S0, s_minus = m.S0;
        for (Eigen::Index j = 1; j <= M; ++j) {
            const double z = gauss(rng);
            s_plus  *= std::exp(drift + vol * z);
            s_minus *= std::exp(drift - vol * z);
            paths.S(2 * p,     j) = s_plus;
            paths.S(2 * p + 1, j) = s_minus;
        }
    }
    return paths;
}

}  // namespace pricer
