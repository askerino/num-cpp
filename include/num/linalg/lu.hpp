#pragma once

#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
#include "num/linalg/detail/triangular.hpp"
#include "num/linalg/detail/validation.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <vector>

namespace num::linalg {

/// @defgroup linalg Linear algebra

/// @defgroup lu LU factorization
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct LuFactorization {
    Matrix<T> lower;
    Matrix<T> upper;
    std::vector<std::size_t> permutation; ///< Row `i` of `PA` is row `permutation[i]` of `A`.
    std::size_t row_swaps{};
};

namespace detail {

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

/// @return The factorization, or an error if @p matrix is singular or contains a non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<LuFactorization<T>, Error> lu_factorize(const Matrix<T>& matrix) {
    detail::require_square_matrix(matrix);

    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    const std::size_t n = matrix.rows();
    Matrix<T> lower(n, n);
    Matrix<T> upper = matrix;
    std::vector<std::size_t> permutation(n);
    for (std::size_t i = 0; i < n; ++i) {
        permutation[i] = i;
    }
    std::size_t row_swaps{};

    for (std::size_t k = 0; k < n; ++k) {
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
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(upper(k, j), upper(pivot_row, j));
                std::swap(lower(k, j), lower(pivot_row, j));
            }
            std::swap(permutation[k], permutation[pivot_row]);
            ++row_swaps;
        }

        lower(k, k) = T{1};
        for (std::size_t i = k + 1; i < n; ++i) {
            const T elimination_factor = upper(i, k) / upper(k, k);
            lower(i, k) = elimination_factor;
            upper(i, k) = T{};
            for (std::size_t j = k + 1; j < n; ++j) {
                upper(i, j) -= elimination_factor * upper(k, j);
            }
        }
    }

    return LuFactorization<T>{lower, upper, permutation, row_swaps};
}

/// @note The product of the diagonal entries may overflow or underflow for large matrices; on
/// underflow the result is indistinguishable from a singular determinant.
template <std::floating_point T>
[[nodiscard]] T lu_determinant(const LuFactorization<T>& factorization) {
    T result = factorization.row_swaps % 2 == 0 ? T{1} : T{-1};
    for (std::size_t i = 0; i < factorization.upper.rows(); ++i) {
        result *= factorization.upper(i, i);
    }
    return result;
}

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

template <std::floating_point T>
[[nodiscard]] Matrix<T> lu_inverse(const LuFactorization<T>& factorization) {
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

/// @}

} // namespace num::linalg
