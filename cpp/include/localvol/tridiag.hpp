#ifndef LOCALVOL_TRIDIAG_HPP
#define LOCALVOL_TRIDIAG_HPP

/// \file tridiag.hpp
/// \brief Native Thomas tridiagonal solver — the single linear-algebra
///        kernel shared by the spline builder and the PDE pricer.
///
/// Solves A x = d where A is tridiagonal with sub-diagonal \c sub (length
/// n-1), diagonal \c diag (length n) and super-diagonal \c sup (length n-1).
/// The Thomas algorithm is Gaussian elimination without pivoting in O(n):
/// it is stable for the diagonally-dominant systems produced by the
/// Crank-Nicolson discretisation and the natural-spline moment equations.
/// The CN systems are diagonally dominant only while both off-diagonals are
/// non-negative, i.e. under the mesh Peclet condition |mu| h <= 2a for
/// central differencing; the PDE switches nodes that violate it to upwind
/// differencing (see pde.hpp) so this kernel never sees a non-M-matrix.
/// No library banded solver is used anywhere in the pricing path.

#include <vector>

namespace localvol {

/// Solve a tridiagonal system by the Thomas algorithm.
///
/// \param sub  sub-diagonal (length n-1); sub[i] multiplies x[i] in row i+1.
/// \param diag main diagonal (length n).
/// \param sup  super-diagonal (length n-1); sup[i] multiplies x[i+1] in row i.
/// \param rhs  right-hand side (length n).
/// \return the solution vector x (length n).
/// \throws std::invalid_argument on inconsistent lengths, an empty system,
///         non-finite input, or a |pivot| < 1e-300 during elimination.
std::vector<double> thomas_solve(const std::vector<double>& sub,
                                 const std::vector<double>& diag,
                                 const std::vector<double>& sup,
                                 const std::vector<double>& rhs);

}  // namespace localvol

#endif  // LOCALVOL_TRIDIAG_HPP
