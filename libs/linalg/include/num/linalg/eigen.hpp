#pragma once

#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
#include "num/linalg/detail/jacobi.hpp"
#include "num/linalg/detail/validation.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <limits>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

namespace num::linalg {

/// @defgroup eigen Eigenvalue decomposition
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct EigenDecomposition {
    Vector<T> values;  ///< Eigenvalues sorted in ascending order.
    Matrix<T> vectors; ///< Orthogonal matrix whose columns are eigenvectors, with the
                       ///< largest-magnitude component of each eigenvector positive.
};

namespace detail::eigen {

inline constexpr int tolerance_epsilon_multiplier = 100;
inline constexpr std::size_t default_max_sweeps = 50;

template <std::floating_point T>
[[nodiscard]] T off_diagonal_norm_l2(const Matrix<T>& matrix) {
    T result{};
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = i + 1; j < matrix.cols(); ++j) {
            result = std::hypot(result, matrix(i, j));
        }
    }
    return result;
}

template <std::floating_point T>
[[nodiscard]] bool is_converged(const Matrix<T>& matrix, T tolerance) {
    return off_diagonal_norm_l2(matrix) <= tolerance * matrix.norm_frobenius();
}

template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void rotate(Matrix<T>& a, Matrix<T>& eigenvectors, std::size_t p, std::size_t q) {
    const T a_pp = a(p, p);
    const T a_qq = a(q, q);
    const T a_pq = a(p, q);
    const auto rotation = jacobi::make_rotation(a_pp, a_qq, a_pq);

    a(p, p) = a_pp - (rotation.tan * a_pq);
    a(q, q) = a_qq + (rotation.tan * a_pq);
    a(p, q) = T{};
    a(q, p) = T{};

    for (std::size_t i = 0; i < a.rows(); ++i) {
        if (i != p && i != q) {
            const T a_ip = a(i, p);
            const T a_iq = a(i, q);
            const T new_a_ip = (rotation.cos * a_ip) - (rotation.sin * a_iq);
            const T new_a_iq = (rotation.sin * a_ip) + (rotation.cos * a_iq);
            a(i, p) = new_a_ip;
            a(p, i) = new_a_ip;
            a(i, q) = new_a_iq;
            a(q, i) = new_a_iq;
        }
    }

    jacobi::rotate_columns(eigenvectors, p, q, rotation);
}

template <std::floating_point T>
void sort_eigenpairs(Vector<T>& values, Matrix<T>& vectors) {
    const std::size_t n = values.size();

    std::vector<std::size_t> order(n);
    std::ranges::iota(order, std::size_t{0});
    std::ranges::stable_sort(order, {}, [&values](std::size_t k) { return values[k]; });

    Vector<T> sorted_values(n);
    Matrix<T> sorted_vectors(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t source = order[k];
        sorted_values[k] = values[source];
        for (std::size_t i = 0; i < n; ++i) {
            sorted_vectors(i, k) = vectors(i, source);
        }
    }

    values = std::move(sorted_values);
    vectors = std::move(sorted_vectors);
}

template <std::floating_point T>
void orient_eigenvectors(Matrix<T>& eigenvectors) {
    const std::size_t n = eigenvectors.rows();
    for (std::size_t k = 0; k < n; ++k) {
        const auto magnitude = [&eigenvectors, k](std::size_t i) {
            return std::abs(eigenvectors(i, k));
        };
        const std::size_t largest =
            *std::ranges::max_element(std::views::iota(std::size_t{0}, n), {}, magnitude);
        if (eigenvectors(largest, k) < T{}) {
            for (std::size_t i = 0; i < n; ++i) {
                eigenvectors(i, k) = -eigenvectors(i, k);
            }
        }
    }
}

} // namespace detail::eigen

/// @return The decomposition, or an error if @p matrix is not symmetric, contains a non-finite
/// value, or the iteration does not converge.
/// @throws std::invalid_argument If @p matrix is empty or non-square, or @p tolerance is not
/// finite or negative.
template <std::floating_point T>
[[nodiscard]] std::expected<EigenDecomposition<T>, Error>
eigen_decompose(const Matrix<T>& matrix,
                T tolerance = static_cast<T>(detail::eigen::tolerance_epsilon_multiplier) *
                              std::numeric_limits<T>::epsilon(),
                std::size_t max_sweeps = detail::eigen::default_max_sweeps) {
    detail::require_valid_tolerance(tolerance);
    detail::require_square_matrix(matrix);
    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }
    if (!matrix.is_symmetric()) {
        return std::unexpected(Error::non_symmetric);
    }

    const std::size_t n = matrix.rows();
    Matrix<T> a = matrix;
    Matrix<T> eigenvectors = Matrix<T>::identity(n);

    std::size_t sweeps = 0;
    while (!detail::eigen::is_converged(a, tolerance)) {
        if (sweeps >= max_sweeps) {
            return std::unexpected(Error::no_convergence);
        }
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (a(p, q) != T{}) {
                    detail::eigen::rotate(a, eigenvectors, p, q);
                }
            }
        }
        ++sweeps;
    }

    Vector<T> eigenvalues(n);
    for (std::size_t k = 0; k < n; ++k) {
        eigenvalues[k] = a(k, k);
    }
    detail::eigen::sort_eigenpairs(eigenvalues, eigenvectors);
    detail::eigen::orient_eigenvectors(eigenvectors);
    return EigenDecomposition<T>{std::move(eigenvalues), std::move(eigenvectors)};
}

/// @}

} // namespace num::linalg
