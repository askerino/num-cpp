#pragma once

#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <format>
#include <stdexcept>
#include <vector>

namespace num::linalg {

/// @defgroup linalg Linear algebra
/// @brief Linear algebra algorithms.
///
/// Formulas are written in plain text with 0-based indices to match the code, e.g. `a_ij` for the
/// element in row `i` and column `j`, and `sum_i` for a sum over all `i`.

/// @defgroup lu LU factorization
/// @ingroup linalg
/// @{

/// @brief Stores an LU factorization satisfying `PA = LU`.
template <std::floating_point T>
struct LuFactorization {
    Matrix<T> lower;                      ///< Unit lower-triangular factor `L`.
    Matrix<T> upper;                      ///< Upper-triangular factor `U`.
    std::vector<std::size_t> permutation; ///< Row `i` of `PA` is row `permutation[i]` of `A`.
    std::size_t row_swaps{};              ///< Number of row exchanges.
};

namespace detail {

template <typename Range>
    requires std::floating_point<std::ranges::range_value_t<Range>>
[[nodiscard]] bool all_finite(const Range& values) {
    return std::ranges::all_of(values, [](const auto value) { return std::isfinite(value); });
}

template <std::floating_point T>
void require_rhs_size(std::size_t system_size, const Vector<T>& rhs) {
    if (rhs.size() != system_size) {
        throw std::invalid_argument(std::format(
            "LU solve size mismatch: system size {}, rhs size {}", system_size, rhs.size()));
    }
}

// Solves Ly = b for lower-triangular L, from the top row down:
// y_i = (b_i - sum_{j<i} l_ij * y_j) / l_ii
template <std::floating_point T>
[[nodiscard]] Vector<T> forward_substitution(const Matrix<T>& lower, const Vector<T>& rhs) {
    const std::size_t n = lower.rows();
    Vector<T> y(n);

    for (std::size_t i = 0; i < n; ++i) {
        T sum{};
        for (std::size_t j = 0; j < i; ++j) {
            sum += lower(i, j) * y[j];
        }
        y[i] = (rhs[i] - sum) / lower(i, i);
    }

    return y;
}

// Solves Ux = y for upper-triangular U, from the bottom row up:
// x_i = (y_i - sum_{j>i} u_ij * x_j) / u_ii
template <std::floating_point T>
[[nodiscard]] Vector<T> backward_substitution(const Matrix<T>& upper, const Vector<T>& rhs) {
    const std::size_t n = upper.rows();
    Vector<T> x(n);

    for (std::size_t i = n; i > 0;) {
        --i;
        T sum{};
        for (std::size_t j = i + 1; j < n; ++j) {
            sum += upper(i, j) * x[j];
        }
        x[i] = (rhs[i] - sum) / upper(i, i);
    }

    return x;
}

// Since PA = LU, Ax = b becomes L(Ux) = Pb, so solve Ly = Pb and then Ux = y.
template <std::floating_point T>
[[nodiscard]] Vector<T> solve_factorized(const LuFactorization<T>& factorization,
                                         const Vector<T>& rhs) {
    const std::size_t n = factorization.lower.rows();
    Vector<T> permuted_rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        permuted_rhs[i] = rhs[factorization.permutation[i]];
    }

    const Vector<T> y = forward_substitution(factorization.lower, permuted_rhs);
    return backward_substitution(factorization.upper, y);
}

} // namespace detail

/// @brief Computes an LU factorization of @p matrix with partial pivoting.
///
/// Starting from `U = A` and `P = I`, for each column `k`:
/// 1. Choose the pivot row `r >= k` with the largest `|u_rk|`,
///    and swap rows `k` and `r` of `U`, `L`, and `P`.
///    If `u_rk = 0`, the matrix is singular.
/// 2. Set `l_kk = 1`.
/// 3. For each row `i > k`, set `l_ik = u_ik / u_kk`,
///    then subtract `l_ik` times row `k` from row `i` of `U`:
///    `u_ij -= l_ik * u_kj` for each column `j >= k`.
///
/// The result satisfies `PA = LU`.
///
/// @return The factorization, or an error if @p matrix is singular or contains a non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
/// @note Only exactly zero pivots are reported as singular; near-singular matrices are not
/// detected.
template <std::floating_point T>
[[nodiscard]] std::expected<LuFactorization<T>, Error> lu_factorize(const Matrix<T>& matrix) {
    const std::size_t n = matrix.rows();
    if (n == 0 || n != matrix.cols()) {
        throw std::invalid_argument(std::format("matrix must be non-empty and square: got {}x{}",
                                                matrix.rows(), matrix.cols()));
    }

    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    Matrix<T> lower(n, n);
    Matrix<T> upper = matrix;
    std::vector<std::size_t> permutation(n);
    for (std::size_t i = 0; i < n; ++i) {
        permutation[i] = i;
    }
    std::size_t row_swaps{};

    for (std::size_t k = 0; k < n; ++k) {
        // Step 1: choose the pivot row and swap rows.
        std::size_t pivot_row = k;
        T pivot_magnitude = std::abs(upper(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            const T candidate_magnitude = std::abs(upper(i, k));
            if (candidate_magnitude > pivot_magnitude) {
                pivot_magnitude = candidate_magnitude;
                pivot_row = i;
            }
        }

        if (pivot_magnitude == T{}) {
            return std::unexpected(Error::singular_matrix);
        }

        if (pivot_row != k) {
            // Columns k and beyond of L are still zero, so entire rows can be swapped.
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(upper(k, j), upper(pivot_row, j));
                std::swap(lower(k, j), lower(pivot_row, j));
            }
            std::swap(permutation[k], permutation[pivot_row]);
            ++row_swaps;
        }

        // Steps 2 and 3: build column k of L and eliminate below the pivot.
        lower(k, k) = T{1};
        for (std::size_t i = k + 1; i < n; ++i) {
            const T elimination_factor = upper(i, k) / upper(k, k);
            lower(i, k) = elimination_factor;
            // u_ik - l_ik * u_kk is zero in exact arithmetic, so set it directly to avoid rounding
            // error.
            upper(i, k) = T{};
            for (std::size_t j = k + 1; j < n; ++j) {
                upper(i, j) -= elimination_factor * upper(k, j);
            }
        }
    }

    return LuFactorization<T>{lower, upper, permutation, row_swaps};
}

/// @brief Computes the determinant of the factorized matrix.
///
/// Since `PA = LU`, `det(L) = 1`, and each row swap flips the sign of the determinant,
/// `det(A) = (-1)^s * prod_i u_ii`, where `s` is the number of row swaps.
template <std::floating_point T>
[[nodiscard]] T determinant(const LuFactorization<T>& factorization) {
    T result = factorization.row_swaps % 2 == 0 ? T{1} : T{-1};
    for (std::size_t i = 0; i < factorization.upper.rows(); ++i) {
        result *= factorization.upper(i, i);
    }
    return result;
}

/// @brief Computes the determinant of @p matrix using LU factorization.
/// @return The determinant, which is zero if @p matrix is singular, or an error if @p matrix
/// contains a non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<T, Error> determinant(const Matrix<T>& matrix) {
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        if (factorization.error() == Error::singular_matrix) {
            return T{};
        }
        return std::unexpected(factorization.error());
    }
    return determinant(*factorization);
}

/// @brief Solves `Ax = b` for @p rhs `b` using @p factorization.
///
/// Since `PA = LU`, it solves `Ly = Pb` by forward substitution and then `Ux = y` by backward
/// substitution.
///
/// @return The solution, or an error if @p rhs contains a non-finite value.
/// @throws std::invalid_argument If @p rhs has an incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, Error> lu_solve(const LuFactorization<T>& factorization,
                                                       const Vector<T>& rhs) {
    detail::require_rhs_size(factorization.lower.rows(), rhs);

    if (!detail::all_finite(rhs)) {
        return std::unexpected(Error::non_finite_input);
    }

    return detail::solve_factorized(factorization, rhs);
}

/// @brief Solves `Ax = b` by computing an LU factorization of @p matrix.
/// @return The solution, or an error if @p matrix is singular or either input contains a non-finite
/// value.
/// @throws std::invalid_argument If @p matrix is empty or non-square, or if @p rhs has an
/// incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, Error> lu_solve(const Matrix<T>& matrix,
                                                       const Vector<T>& rhs) {
    detail::require_rhs_size(matrix.rows(), rhs);
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        return std::unexpected(factorization.error());
    }
    return lu_solve(*factorization, rhs);
}

/// @brief Computes the inverse of the factorized matrix.
///
/// Column `j` of `A^-1` is the solution of `Ax = e_j`, where `e_j` is the `j`-th standard basis
/// vector.
template <std::floating_point T>
[[nodiscard]] Matrix<T> inverse(const LuFactorization<T>& factorization) {
    const std::size_t n = factorization.lower.rows();
    Matrix<T> result(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        Vector<T> basis_vector(n);
        basis_vector[j] = T{1};
        const Vector<T> inverse_column = detail::solve_factorized(factorization, basis_vector);
        for (std::size_t i = 0; i < n; ++i) {
            result(i, j) = inverse_column[i];
        }
    }

    return result;
}

/// @brief Computes the inverse of @p matrix using LU factorization.
/// @return The inverse, or an error if @p matrix is singular or contains a non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<Matrix<T>, Error> inverse(const Matrix<T>& matrix) {
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        return std::unexpected(factorization.error());
    }
    return inverse(*factorization);
}

/// @}

} // namespace num::linalg
