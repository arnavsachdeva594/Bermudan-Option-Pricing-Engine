#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "gbm.hpp"
#include "params.hpp"
#include "payoff.hpp"

namespace pricer {

// ---------------------------------------------------------------------------
// Pluggable regression basis.
//
// A Basis turns a spot price S into a feature row phi(S) = [f0(S), f1(S), ...].
// Longstaff-Schwartz regresses realized continuation values onto these
// features to estimate the conditional continuation value at each date.
// ---------------------------------------------------------------------------
class Basis {
public:
    virtual ~Basis() = default;
    // Number of basis functions (columns of the design matrix).
    [[nodiscard]] virtual int size() const = 0;
    // Fill out[0..size()-1] with the feature values at spot S.
    virtual void eval(double S, double* out) const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

// Monomials {1, S, S^2, ..., S^degree}. degree=2 gives the spec basis {1,S,S^2}.
class PolynomialBasis final : public Basis {
public:
    explicit PolynomialBasis(int degree = 2) : degree_(degree) {}
    [[nodiscard]] int size() const override { return degree_ + 1; }
    void eval(double S, double* out) const override {
        double p = 1.0;
        for (int k = 0; k <= degree_; ++k) {
            out[k] = p;
            p *= S;
        }
    }
    [[nodiscard]] std::string name() const override {
        return "Polynomial(deg=" + std::to_string(degree_) + ")";
    }

private:
    int degree_;
};

// Weighted-free Laguerre polynomials L_0..L_degree evaluated on moneyness
// x = S/strike (keeps magnitudes O(1) so the polynomials stay well-behaved).
// Recurrence: L_{k+1}(x) = ((2k+1 - x) L_k(x) - k L_{k-1}(x)) / (k+1).
class LaguerreBasis final : public Basis {
public:
    LaguerreBasis(double strike, int degree = 3) : strike_(strike), degree_(degree) {}
    [[nodiscard]] int size() const override { return degree_ + 1; }
    void eval(double S, double* out) const override {
        const double x = S / strike_;
        out[0] = 1.0;
        if (degree_ >= 1) out[1] = 1.0 - x;
        for (int k = 1; k < degree_; ++k) {
            out[k + 1] =
                ((2.0 * k + 1.0 - x) * out[k] - static_cast<double>(k) * out[k - 1]) /
                static_cast<double>(k + 1);
        }
    }
    [[nodiscard]] std::string name() const override {
        return "Laguerre(deg=" + std::to_string(degree_) + ")";
    }

private:
    double strike_;
    int    degree_;
};

// ---------------------------------------------------------------------------
// LSM result: the price estimate and its Monte-Carlo standard error.
// ---------------------------------------------------------------------------
struct LSMResult {
    double price = 0.0;
    double std_error = 0.0;
};

// Price a Bermudan option by least-squares Monte Carlo on already-generated
// paths. Exercise dates are the M steps t_1..t_M of `paths` (t_M = maturity).
// Regression uses only in-the-money paths and is solved via householderQr.
LSMResult longstaff_schwartz(const Paths& paths,
                             const Payoff& payoff,
                             const MarketParams& m,
                             const Basis& basis);

}  // namespace pricer
