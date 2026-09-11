// Benchmark harness for the Bermudan option pricing engine.
//
// Prints a table of  paths | price | std error | se*sqrt(N) | wall time  for
// three estimators (plain MC, antithetic, antithetic + control variate) across
// a range of path counts. Two things to read off the table:
//   * Down a column, se*sqrt(N) is roughly constant -> SE falls as 1/sqrt(N).
//   * Across methods at fixed N, the standard error shrinks -> variance
//     reduction is working.
//
// Usage:
//   ./benchmark            run 1e3, 1e4, 1e5, 1e6 paths (default)
//   ./benchmark 5          run up to 1e5 paths (max power of ten = 5)
//   ./benchmark 4 7        run 1e4 .. 1e7 paths (min power, max power)

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "blackscholes.hpp"
#include "binomial.hpp"
#include "lsm.hpp"
#include "params.hpp"
#include "payoff.hpp"

using namespace pricer;

namespace {

struct Row {
    std::string method;
    std::size_t paths;
    double      price;
    double      std_error;
    double      wall_ms;
};

Row run_one(const std::string& label, const MarketParams& m, MCConfig cfg,
            const Payoff& payoff, const Basis& basis, VarianceReduction vr) {
    const auto t0 = std::chrono::steady_clock::now();
    const Estimate e = price_bermudan(m, cfg, payoff, basis, vr);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return Row{label, cfg.num_paths, e.price, e.std_error, ms};
}

void print_header() {
    std::cout << std::left << std::setw(22) << "method"
              << std::right << std::setw(12) << "paths"
              << std::setw(12) << "price"
              << std::setw(14) << "std error"
              << std::setw(14) << "se*sqrt(N)"
              << std::setw(14) << "wall (ms)" << "\n";
    std::cout << std::string(88, '-') << "\n";
}

void print_row(const Row& r) {
    const double se_root_n = r.std_error * std::sqrt(static_cast<double>(r.paths));
    std::cout << std::left << std::setw(22) << r.method
              << std::right << std::setw(12) << r.paths
              << std::setw(12) << std::fixed << std::setprecision(4) << r.price
              << std::setw(14) << std::setprecision(6) << r.std_error
              << std::setw(14) << std::setprecision(4) << se_root_n
              << std::setw(14) << std::setprecision(1) << r.wall_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    int min_pow = 3, max_pow = 6;
    if (argc == 2) max_pow = std::atoi(argv[1]);
    if (argc >= 3) { min_pow = std::atoi(argv[1]); max_pow = std::atoi(argv[2]); }

    MarketParams m;                       // spec defaults: S0=K=100, r=5%, sig=20%, T=1
    Payoff       put{OptionType::Put, m.K};
    PolynomialBasis basis(2);             // {1, S, S^2}

    // Reference prices for context (printed once, above the table).
    const double euro = black_scholes_price(put, m);
    const double amer = binomial_price(put, m, 5000, /*american=*/true);

    std::cout << "Bermudan put, S0=" << m.S0 << " K=" << m.K << " r=" << m.r
              << " sigma=" << m.sigma << " T=" << m.T
              << ", " << 50 << " exercise dates\n";
    std::cout << "hardware_concurrency = " << std::thread::hardware_concurrency()
              << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "European (Black-Scholes) = " << euro
              << "   |   American (tree, 5000 steps) = " << amer << "\n";
    std::cout << "Bermudan price must sit between these two.\n\n";

    const struct { const char* label; VarianceReduction vr; } methods[] = {
        {"plain MC",              {false, false}},
        {"antithetic",            {true,  false}},
        {"antithetic + control",  {true,  true }},
    };

    print_header();
    for (int p = min_pow; p <= max_pow; ++p) {
        std::size_t N = 1;
        for (int k = 0; k < p; ++k) N *= 10;
        for (const auto& method : methods) {
            MCConfig cfg;
            cfg.num_paths = N;
            cfg.num_steps = 50;
            cfg.seed = 20240101ULL;       // fixed master seed
            cfg.num_threads = 0;          // auto
            print_row(run_one(method.label, m, cfg, put, basis, method.vr));
        }
        std::cout << "\n";
    }
    return 0;
}
