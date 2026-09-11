#pragma once

#include <algorithm>
#include <string>

namespace pricer {

// A vanilla option is either a Call or a Put. The only thing that differs
// between them is the shape of the intrinsic (immediate-exercise) value.
enum class OptionType { Call, Put };

// The Payoff is the *contract terminal condition*: given a spot price S,
// what does exercising the option right now pay?
//
//   Put : max(K - S, 0)   -- right to SELL at K, worth it when S < K
//   Call: max(S - K, 0)   -- right to BUY  at K, worth it when S > K
//
// The max(.,0) encodes "right, not obligation": you never exercise into a loss.
//
// This is deliberately a tiny value type. Every pricer (Black-Scholes, the
// binomial tree, Longstaff-Schwartz) asks the Payoff for intrinsic value, so
// the same pricing code handles calls and puts with no special-casing.
struct Payoff {
    OptionType type = OptionType::Put;
    double     strike = 100.0;

    // Intrinsic value at spot S: what exercising right now earns.
    [[nodiscard]] double intrinsic(double S) const {
        return type == OptionType::Call ? std::max(S - strike, 0.0)
                                        : std::max(strike - S, 0.0);
    }

    // "In the money" means exercising now pays something strictly positive.
    // Longstaff-Schwartz regresses ONLY on ITM paths, so this predicate gets
    // used directly by the LSM core in a later stage.
    [[nodiscard]] bool in_the_money(double S) const {
        return intrinsic(S) > 0.0;
    }
};

inline std::string to_string(OptionType t) {
    return t == OptionType::Call ? "Call" : "Put";
}

}  // namespace pricer
