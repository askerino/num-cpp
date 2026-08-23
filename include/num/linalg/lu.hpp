#pragma once

#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
#include "num/linalg/error.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <format>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Stores an LU factorization satisfying `PA = LU`.
/// @note `permutation[i]` gives the original row of `A` placed at row `i` of `PA`.
template <std::floating_point T>
struct LuFactorization {
    Matrix<T> lower;                      ///< Unit lower-triangular factor L.
    Matrix<T> upper;                      ///< Upper-triangular factor U.
    std::vector<std::size_t> permutation; ///< Row permutation vector representing P.
    std::size_t row_swaps{};              ///< Number of row exchanges.
};

namespace detail {

template <typename Range>
    requires std::floating_point<std::ranges::range_value_t<Range>>
[[nodiscard]] bool all_finite(const Range& values) {
    return std::ranges::all_of(values, [](const auto value) { return std::isfinite(value); });
}

template <std::floating_point T>
[[nodiscard]] Vector<T> forward_substitution(const Matrix<T>& lower, const Vector<T>& rhs) {
    const std::size_t system_size = lower.rows();
    Vector<T> result(system_size);

    for (std::size_t row = 0; row < system_size; ++row) {
        T sum{};
        for (std::size_t column = 0; column < row; ++column) {
            sum += lower(row, column) * result[column];
        }
        result[row] = (rhs[row] - sum) / lower(row, row);
    }

    return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> backward_substitution(const Matrix<T>& upper, const Vector<T>& rhs) {
    const std::size_t system_size = upper.rows();
    Vector<T> result(system_size);

    for (std::size_t row = system_size; row > 0;) {
        --row;
        T sum{};
        for (std::size_t column = row + 1; column < system_size; ++column) {
            sum += upper(row, column) * result[column];
        }
        result[row] = (rhs[row] - sum) / upper(row, row);
    }

    return result;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> solve_factorized(const LuFactorization<T>& factorization,
                                         const Vector<T>& rhs) {
    const std::size_t system_size = factorization.lower.rows();
    Vector<T> permuted_rhs(system_size);
    for (std::size_t index = 0; index < system_size; ++index) {
        permuted_rhs[index] = rhs[factorization.permutation[index]];
    }

    const Vector<T> forward_solution = forward_substitution(factorization.lower, permuted_rhs);
    return backward_substitution(factorization.upper, forward_solution);
}

} // namespace detail

/// @brief Computes an LU factorization of @p matrix with partial pivoting.
///
/// Produces factors and a permutation satisfying `PA = LU`.
///
/// @return The factorization, or an error if @p matrix is singular or contains a non-finite value.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
/// @note Singularity is detected only when the magnitude of the selected pivot is exactly zero;
/// near-singular matrices are not diagnosed separately.
template <std::floating_point T>
[[nodiscard]] std::expected<LuFactorization<T>, LinalgError> lu_factorize(const Matrix<T>& matrix) {
    const std::size_t system_size = matrix.rows();
    if (system_size == 0 || system_size != matrix.cols()) {
        throw std::invalid_argument(std::format("matrix must be non-empty and square: got {}x{}",
                                                matrix.rows(), matrix.cols()));
    }

    if (!detail::all_finite(matrix)) {
        return std::unexpected(LinalgError::non_finite_input);
    }

    Matrix<T> lower(system_size, system_size);
    Matrix<T> upper = matrix;
    std::vector<std::size_t> permutation(system_size);
    for (std::size_t index = 0; index < system_size; ++index) {
        lower(index, index) = T{1};
        permutation[index] = index;
    }
    std::size_t row_swaps{};

    for (std::size_t pivot_index = 0; pivot_index < system_size; ++pivot_index) {
        // Select the largest absolute value in the current column as the pivot.
        std::size_t pivot_row = pivot_index;
        T pivot_magnitude = std::abs(upper(pivot_index, pivot_index));
        for (std::size_t row = pivot_index + 1; row < system_size; ++row) {
            const T candidate_magnitude = std::abs(upper(row, pivot_index));
            if (candidate_magnitude > pivot_magnitude) {
                pivot_magnitude = candidate_magnitude;
                pivot_row = row;
            }
        }

        if (pivot_magnitude == T{}) {
            return std::unexpected(LinalgError::singular_matrix);
        }

        if (pivot_row != pivot_index) {
            for (std::size_t swap_column = 0; swap_column < system_size; ++swap_column) {
                std::swap(upper(pivot_index, swap_column), upper(pivot_row, swap_column));
            }
            for (std::size_t swap_column = 0; swap_column < pivot_index; ++swap_column) {
                std::swap(lower(pivot_index, swap_column), lower(pivot_row, swap_column));
            }
            std::swap(permutation[pivot_index], permutation[pivot_row]);
            ++row_swaps;
        }

        for (std::size_t row = pivot_index + 1; row < system_size; ++row) {
            // Eliminate U(row, pivot_index) and store the multiplier in L.
            const T elimination_factor = upper(row, pivot_index) / upper(pivot_index, pivot_index);
            lower(row, pivot_index) = elimination_factor;
            upper(row, pivot_index) = T{};
            for (std::size_t trailing_column = pivot_index + 1; trailing_column < system_size;
                 ++trailing_column) {
                upper(row, trailing_column) -=
                    elimination_factor * upper(pivot_index, trailing_column);
            }
        }
    }

    return LuFactorization<T>{lower, upper, permutation, row_swaps};
}

/// @brief Computes the determinant from @p factorization.
/// @pre @p factorization is valid and satisfies `PA = LU`.
template <std::floating_point T>
[[nodiscard]] T determinant(const LuFactorization<T>& factorization) {
    T result = factorization.row_swaps % 2 == 0 ? T{1} : T{-1};
    for (std::size_t index = 0; index < factorization.upper.rows(); ++index) {
        result *= factorization.upper(index, index);
    }
    return result;
}

/// @brief Computes the determinant of @p matrix using LU factorization.
/// @return The determinant, or an error for non-finite input.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<T, LinalgError> determinant(const Matrix<T>& matrix) {
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        if (factorization.error() == LinalgError::singular_matrix) {
            return T{};
        }
        return std::unexpected(factorization.error());
    }
    return determinant(*factorization);
}

/// @brief Solves `Ax = b` using @p factorization and the right-hand side @p rhs.
/// @pre @p factorization is valid and satisfies `PA = LU`.
/// @return The solution, or an error if @p rhs contains a non-finite value.
/// @throws std::invalid_argument If @p rhs has an incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, LinalgError>
lu_solve(const LuFactorization<T>& factorization, const Vector<T>& rhs) {
    const std::size_t system_size = factorization.lower.rows();
    if (rhs.size() != system_size) {
        throw std::invalid_argument(std::format(
            "LU solve size mismatch: system size {}, rhs size {}", system_size, rhs.size()));
    }

    if (!detail::all_finite(rhs)) {
        return std::unexpected(LinalgError::non_finite_input);
    }

    return detail::solve_factorized(factorization, rhs);
}

/// @brief Solves `Ax = b` by computing an LU factorization of @p matrix.
/// @return The solution, or an error for a singular matrix or non-finite input.
/// @throws std::invalid_argument If @p matrix is empty or non-square, or if @p rhs has an
/// incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, LinalgError> lu_solve(const Matrix<T>& matrix,
                                                             const Vector<T>& rhs) {
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        return std::unexpected(factorization.error());
    }
    return lu_solve(*factorization, rhs);
}

/// @brief Computes the inverse matrix from @p factorization.
/// @pre @p factorization is valid and satisfies `PA = LU`.
template <std::floating_point T>
[[nodiscard]] Matrix<T> inverse(const LuFactorization<T>& factorization) {
    const std::size_t system_size = factorization.lower.rows();
    Matrix<T> result(system_size, system_size);

    for (std::size_t column = 0; column < system_size; ++column) {
        // Solve Ax = e_i to compute column i of the inverse.
        Vector<T> basis_vector(system_size);
        basis_vector[column] = T{1};
        const Vector<T> inverse_column = detail::solve_factorized(factorization, basis_vector);
        for (std::size_t row = 0; row < system_size; ++row) {
            result(row, column) = inverse_column[row];
        }
    }

    return result;
}

/// @brief Computes the inverse of @p matrix using LU factorization.
/// @return The inverse, or an error for a singular matrix or non-finite input.
/// @throws std::invalid_argument If @p matrix is empty or non-square.
template <std::floating_point T>
[[nodiscard]] std::expected<Matrix<T>, LinalgError> inverse(const Matrix<T>& matrix) {
    const auto factorization = lu_factorize(matrix);
    if (!factorization) {
        return std::unexpected(factorization.error());
    }
    return inverse(*factorization);
}

} // namespace num
