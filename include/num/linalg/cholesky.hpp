#pragma once

#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
#include "num/linalg/detail/triangular.hpp"
#include "num/linalg/detail/validation.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>

namespace num::linalg {

/// @defgroup cholesky Cholesky factorization
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct CholeskyFactorization {
    Matrix<T> lower;
};

namespace detail {

template <std::floating_point T>
[[nodiscard]] Vector<T> solve_factorized(const CholeskyFactorization<T>& factorization,
                                         const Vector<T>& rhs) {
    const Vector<T> y = forward_substitution(factorization.lower, rhs);
    return backward_substitution(factorization.lower.transpose(), y);
}

} // namespace detail

/// @return The factorization, or an error if @p matrix is not positive definite or contains a
/// non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<CholeskyFactorization<T>, Error>
cholesky_factorize(const Matrix<T>& matrix) {
    detail::require_square_matrix(matrix);

    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    const std::size_t n = matrix.rows();
    Matrix<T> lower(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        T diagonal = matrix(j, j);
        for (std::size_t k = 0; k < j; ++k) {
            diagonal -= lower(j, k) * lower(j, k);
        }
        if (diagonal <= T{}) {
            return std::unexpected(Error::non_positive_definite);
        }
        lower(j, j) = std::sqrt(diagonal);

        for (std::size_t i = j + 1; i < n; ++i) {
            T value = matrix(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                value -= lower(i, k) * lower(j, k);
            }
            lower(i, j) = value / lower(j, j);
        }
    }

    return CholeskyFactorization<T>{lower};
}

/// @note The product of the squared diagonal entries may overflow or underflow for large matrices;
/// on underflow the result is indistinguishable from a singular determinant.
template <std::floating_point T>
[[nodiscard]] T cholesky_determinant(const CholeskyFactorization<T>& factorization) {
    T result{1};
    for (std::size_t i = 0; i < factorization.lower.rows(); ++i) {
        result *= factorization.lower(i, i) * factorization.lower(i, i);
    }
    return result;
}

/// @return The solution, or an error if @p rhs contains a non-finite value.
/// @throws std::invalid_argument If @p rhs has an incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, Error>
cholesky_solve(const CholeskyFactorization<T>& factorization, const Vector<T>& rhs) {
    detail::require_rhs_size(factorization.lower.rows(), rhs);

    if (!detail::all_finite(rhs)) {
        return std::unexpected(Error::non_finite_input);
    }

    return detail::solve_factorized(factorization, rhs);
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> cholesky_inverse(const CholeskyFactorization<T>& factorization) {
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
