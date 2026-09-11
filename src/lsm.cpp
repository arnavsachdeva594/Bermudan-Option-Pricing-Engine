#include "lsm.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace pricer {

LSMResult longstaff_schwartz(const Paths& paths,
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

    // Discount each path's realized cashflow to today and average.
    double sum = 0.0, sumsq = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        const double pv = cf_amount[static_cast<std::size_t>(i)] *
                          std::pow(df, static_cast<double>(cf_step[static_cast<std::size_t>(i)]));
        sum += pv;
        sumsq += pv * pv;
    }
    const double n = static_cast<double>(N);
    const double mean = sum / n;
    const double var = (sumsq / n - mean * mean) * n / (n - 1.0);  // unbiased

    LSMResult res;
    res.price = mean;
    res.std_error = std::sqrt(var / n);
    return res;
}

}  // namespace pricer
