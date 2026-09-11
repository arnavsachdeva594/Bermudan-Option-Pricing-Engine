#include "lsm.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "blackscholes.hpp"

namespace pricer {

std::vector<double> lsm_pathwise_pv(const Paths& paths,
                                    const Payoff& payoff,
                                    const MarketParams& m,
                                    const Basis& basis) {
    const Eigen::Index N = paths.num_paths();
    const Eigen::Index M = paths.num_steps();
    const int          k = basis.size();

    const double df = std::exp(-m.r * paths.dt);  // one-step discount factor

    // Each path exercises exactly once. Track the cash it receives and when.
    // Init at maturity: value = payoff(S_T), received at step M.
    std::vector<double>        cf_amount(static_cast<std::size_t>(N));
    std::vector<Eigen::Index>  cf_step(static_cast<std::size_t>(N), M);
    for (Eigen::Index i = 0; i < N; ++i) {
        cf_amount[static_cast<std::size_t>(i)] = payoff.intrinsic(paths.S(i, M));
    }

    // Reusable buffers for the per-date regression.
    std::vector<Eigen::Index> itm;                 // indices of ITM paths
    itm.reserve(static_cast<std::size_t>(N));
    std::vector<double>       feat(static_cast<std::size_t>(k));

    // Backward induction over the interior exercise dates t = M-1 .. 1.
    // (t=0 is today: not an exercise date, just the final discounting.)
    for (Eigen::Index t = M - 1; t >= 1; --t) {
        // 1) Collect in-the-money paths at this date.
        itm.clear();
        for (Eigen::Index i = 0; i < N; ++i) {
            if (payoff.intrinsic(paths.S(i, t)) > 0.0) itm.push_back(i);
        }
        const Eigen::Index n_itm = static_cast<Eigen::Index>(itm.size());

        // Not enough data to fit a full-rank system: hold everyone at this date.
        if (n_itm < static_cast<Eigen::Index>(k)) continue;

        // 2) Build the least-squares system A*beta ~= Y over ITM paths.
        //    Y_i = realized continuation value discounted back to date t.
        Eigen::MatrixXd A(n_itm, k);
        Eigen::VectorXd Y(n_itm);
        for (Eigen::Index row = 0; row < n_itm; ++row) {
            const Eigen::Index i = itm[static_cast<std::size_t>(row)];
            basis.eval(paths.S(i, t), feat.data());
            for (int c = 0; c < k; ++c) A(row, c) = feat[static_cast<std::size_t>(c)];

            const Eigen::Index steps_ahead =
                cf_step[static_cast<std::size_t>(i)] - t;
            Y(row) = cf_amount[static_cast<std::size_t>(i)] *
                     std::pow(df, static_cast<double>(steps_ahead));
        }

        // 3) Solve for the continuation-value coefficients (stable QR).
        const Eigen::VectorXd beta = A.householderQr().solve(Y);

        // 4) Exercise where immediate intrinsic beats the fitted continuation.
        for (Eigen::Index row = 0; row < n_itm; ++row) {
            const Eigen::Index i = itm[static_cast<std::size_t>(row)];
            const double intrinsic = payoff.intrinsic(paths.S(i, t));
            const double cont_hat = A.row(row).dot(beta);
            if (intrinsic > cont_hat) {
                cf_amount[static_cast<std::size_t>(i)] = intrinsic;
                cf_step[static_cast<std::size_t>(i)] = t;
            }
        }
    }

    // Discount each path's realized cashflow to today; return the pathwise PVs.
    std::vector<double> pv(static_cast<std::size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i) {
        pv[static_cast<std::size_t>(i)] =
            cf_amount[static_cast<std::size_t>(i)] *
            std::pow(df, static_cast<double>(cf_step[static_cast<std::size_t>(i)]));
    }
    return pv;
}

LSMResult longstaff_schwartz(const Paths& paths,
                             const Payoff& payoff,
                             const MarketParams& m,
                             const Basis& basis) {
    const std::vector<double> pv = lsm_pathwise_pv(paths, payoff, m, basis);
    const Estimate e = reduce(pv, Antithetic::Off);
    return LSMResult{e.price, e.std_error};
}

// --- Variance-reduction estimator ------------------------------------------

namespace {
// Sample mean and unbiased-variance standard error of a vector of values.
Estimate mean_and_se(const std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    double sum = 0.0;
    for (double v : x) sum += v;
    const double mean = sum / n;
    double ss = 0.0;
    for (double v : x) ss += (v - mean) * (v - mean);
    const double var = ss / (n - 1.0);
    return Estimate{mean, std::sqrt(var / n)};
}
}  // namespace

Estimate reduce(const std::vector<double>& pv,
                Antithetic antithetic,
                const std::vector<double>* control,
                double control_mean) {
    // 1) Fold antithetic pairs into their averages: the pair is the
    //    independent sampling unit, so statistics counts pairs, not paths.
    std::vector<double> y;
    std::vector<double> c;
    if (antithetic == Antithetic::On) {
        const std::size_t P = pv.size() / 2;
        y.resize(P);
        for (std::size_t p = 0; p < P; ++p) y[p] = 0.5 * (pv[2 * p] + pv[2 * p + 1]);
        if (control) {
            c.resize(P);
            for (std::size_t p = 0; p < P; ++p)
                c[p] = 0.5 * ((*control)[2 * p] + (*control)[2 * p + 1]);
        }
    } else {
        y = pv;
        if (control) c = *control;
    }

    // 2) No control variate: plain mean + SE of the (possibly folded) units.
    if (!control) return mean_and_se(y);

    // 3) Control variate: Y* = Y - beta*(C - E[C]), beta = Cov(Y,C)/Var(C).
    const double n = static_cast<double>(y.size());
    double ybar = 0.0, cbar = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) { ybar += y[i]; cbar += c[i]; }
    ybar /= n; cbar /= n;

    double cov = 0.0, varc = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double dy = y[i] - ybar, dc = c[i] - cbar;
        cov += dy * dc;
        varc += dc * dc;
    }
    const double beta = (varc > 0.0) ? cov / varc : 0.0;

    std::vector<double> adjusted(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        adjusted[i] = y[i] - beta * (c[i] - control_mean);
    return mean_and_se(adjusted);
}

Estimate price_bermudan(const MarketParams& m,
                        const MCConfig& cfg,
                        const Payoff& payoff,
                        const Basis& basis,
                        VarianceReduction vr) {
    const Paths paths = vr.antithetic ? generate_gbm_paths_antithetic(m, cfg)
                                      : generate_gbm_paths(m, cfg);
    const std::vector<double> pv = lsm_pathwise_pv(paths, payoff, m, basis);

    if (!vr.control) {
        return reduce(pv, vr.antithetic ? Antithetic::On : Antithetic::Off);
    }

    // European control variate: the same option exercised only at maturity.
    // C_i = e^{-rT} * intrinsic(S_T_i); E[C] = Black-Scholes price (exact).
    const double disc_T = std::exp(-m.r * m.T);
    std::vector<double> control(pv.size());
    for (std::size_t i = 0; i < pv.size(); ++i) {
        control[i] = disc_T * payoff.intrinsic(paths.terminal()(static_cast<Eigen::Index>(i)));
    }
    const double control_mean = black_scholes_price(payoff, m);

    return reduce(pv, vr.antithetic ? Antithetic::On : Antithetic::Off,
                  &control, control_mean);
}

}  // namespace pricer
