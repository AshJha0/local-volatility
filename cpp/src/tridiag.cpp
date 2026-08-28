#include "localvol/tridiag.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace localvol {

namespace {
constexpr double kPivotTol = 1e-300;

bool all_finite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}
}  // namespace

std::vector<double> thomas_solve(const std::vector<double>& sub,
                                 const std::vector<double>& diag,
                                 const std::vector<double>& sup,
                                 const std::vector<double>& rhs) {
    const std::size_t n = diag.size();
    if (n == 0) {
        throw std::invalid_argument("thomas_solve: empty system");
    }
    if (sub.size() != n - 1 || sup.size() != n - 1 || rhs.size() != n) {
        throw std::invalid_argument("thomas_solve: inconsistent shapes");
    }
    if (!(all_finite(sub) && all_finite(diag) && all_finite(sup) && all_finite(rhs))) {
        throw std::invalid_argument("thomas_solve: non-finite input");
    }

    // Forward elimination: cp/dp are the modified super-diagonal and rhs.
    std::vector<double> cp(n, 0.0);
    std::vector<double> dp(n, 0.0);
    double piv = diag[0];
    if (std::abs(piv) < kPivotTol) {
        throw std::invalid_argument("thomas_solve: zero pivot at row 0");
    }
    cp[0] = (n > 1) ? sup[0] / piv : 0.0;
    dp[0] = rhs[0] / piv;
    for (std::size_t i = 1; i < n; ++i) {
        piv = diag[i] - sub[i - 1] * cp[i - 1];
        if (std::abs(piv) < kPivotTol) {
            throw std::invalid_argument("thomas_solve: zero pivot at row " + std::to_string(i));
        }
        cp[i] = (i < n - 1) ? sup[i] / piv : 0.0;
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / piv;
    }

    // Back substitution.
    std::vector<double> x(n);
    x[n - 1] = dp[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

}  // namespace localvol
