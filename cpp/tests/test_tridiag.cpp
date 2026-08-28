// Thomas solver: agreement with a dense Gaussian-elimination solve on random
// diagonally-dominant tridiagonal systems, plus input validation.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "localvol/tridiag.hpp"

namespace {

// Dense Gaussian elimination with partial pivoting — an independent check
// (library banded solvers are only allowed inside tests, per the spec).
std::vector<double> dense_solve(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < n; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        }
        std::swap(a[col], a[piv]);
        std::swap(b[col], b[piv]);
        for (std::size_t r = col + 1; r < n; ++r) {
            const double f = a[r][col] / a[col][col];
            for (std::size_t c = col; c < n; ++c) a[r][c] -= f * a[col][c];
            b[r] -= f * b[col];
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double acc = b[i];
        for (std::size_t c = i + 1; c < n; ++c) acc -= a[i][c] * x[c];
        x[i] = acc / a[i][i];
    }
    return x;
}

}  // namespace

TEST(Tridiag, MatchesDenseSolveOnRandomSystems) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {
        const std::size_t n = 2 + static_cast<std::size_t>(trial) % 30;
        std::vector<double> sub(n - 1), diag(n), sup(n - 1), rhs(n);
        for (std::size_t i = 0; i < n - 1; ++i) {
            sub[i] = u(rng);
            sup[i] = u(rng);
        }
        for (std::size_t i = 0; i < n; ++i) {
            // Diagonally dominant: safe for pivot-free Thomas.
            diag[i] = 4.0 + std::abs(u(rng));
            rhs[i] = u(rng);
        }
        std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));
        for (std::size_t i = 0; i < n; ++i) {
            a[i][i] = diag[i];
            if (i > 0) a[i][i - 1] = sub[i - 1];
            if (i + 1 < n) a[i][i + 1] = sup[i];
        }
        const std::vector<double> x = localvol::thomas_solve(sub, diag, sup, rhs);
        const std::vector<double> xd = dense_solve(a, rhs);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(x[i], xd[i], 1e-12) << "trial " << trial << " row " << i;
        }
    }
}

TEST(Tridiag, SolvesSizeOneSystem) {
    const std::vector<double> x = localvol::thomas_solve({}, {2.0}, {}, {5.0});
    ASSERT_EQ(x.size(), 1u);
    EXPECT_DOUBLE_EQ(x[0], 2.5);
}

TEST(Tridiag, RejectsBadInput) {
    EXPECT_THROW(localvol::thomas_solve({}, {}, {}, {}), std::invalid_argument);
    EXPECT_THROW(localvol::thomas_solve({1.0}, {1.0, 1.0, 1.0}, {1.0}, {1.0, 1.0, 1.0}),
                 std::invalid_argument);  // sub/sup length mismatch
    EXPECT_THROW(localvol::thomas_solve({1.0}, {1.0, 1.0}, {1.0}, {1.0}),
                 std::invalid_argument);  // rhs length mismatch
    const double nan = std::nan("");
    EXPECT_THROW(localvol::thomas_solve({nan}, {1.0, 1.0}, {0.0}, {1.0, 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(localvol::thomas_solve({0.0}, {0.0, 1.0}, {0.0}, {1.0, 1.0}),
                 std::invalid_argument);  // zero pivot
}
