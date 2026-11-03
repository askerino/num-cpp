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
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num::linalg {

/// @defgroup svd Singular value decomposition
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct SvdDecomposition {
    /// @brief Matrix whose orthonormal columns are the left singular vectors, with the
    /// largest-magnitude component of each column positive.
    Matrix<T> u;
    /// @brief Singular values sorted in descending order.
    Vector<T> s;
    /// @brief Matrix whose orthonormal columns are the right singular vectors, with each column's
    /// sign chosen in tandem with @c u.
    Matrix<T> v;
};

namespace detail::svd {

inline constexpr int tolerance_epsilon_multiplier = 100;
inline constexpr std::size_t default_max_sweeps = 50;

template <std::floating_point T>
[[nodiscard]] T max_abs(const Matrix<T>& matrix) {
    T result{};
    for (const T value : matrix) {
        const T magnitude = std::abs(value);
        if (magnitude > result) {
            result = magnitude;
        }
    }
    return result;
}

template <std::floating_point T>
struct ColumnGram {
    T alpha; ///< Squared norm of the first column.
    T beta;  ///< Squared norm of the second column.
    T gamma; ///< Dot product of the two columns.
};

template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] ColumnGram<T> column_gram(const Matrix<T>& matrix, std::size_t p, std::size_t q) {
    ColumnGram<T> gram{};
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        const T p_value = matrix(i, p);
        const T q_value = matrix(i, q);
        gram.alpha += p_value * p_value;
        gram.beta += q_value * q_value;
        gram.gamma += p_value * q_value;
    }
    return gram;
}

template <std::floating_point T>
[[nodiscard]] bool is_converged(const Matrix<T>& matrix, T tolerance) {
    for (std::size_t p = 0; p < matrix.cols(); ++p) {
        for (std::size_t q = p + 1; q < matrix.cols(); ++q) {
            const ColumnGram<T> gram = column_gram(matrix, p, q);
            if (std::abs(gram.gamma) > tolerance * std::sqrt(gram.alpha * gram.beta)) {
                return false;
            }
        }
    }
    return true;
}

template <std::floating_point T>
[[nodiscard]] T column_norm(const Matrix<T>& matrix, std::size_t column) {
    T norm{};
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        norm = std::hypot(norm, matrix(i, column));
    }
    return norm;
}

template <std::floating_point T>
void fill_orthonormal_columns(Matrix<T>& u) {
    const std::size_t rows = u.rows();
    const std::size_t columns = u.cols();

    std::vector<bool> has_direction(columns, false);
    for (std::size_t column = 0; column < columns; ++column) {
        has_direction[column] = column_norm(u, column) > T{};
    }

    for (std::size_t column = 0; column < columns; ++column) {
        if (has_direction[column]) {
            continue;
        }

        Vector<T> best_candidate(rows);
        T best_candidate_norm{};
        for (std::size_t basis_index = 0; basis_index < rows; ++basis_index) {
            Vector<T> candidate(rows);
            candidate[basis_index] = T{1};
            for (std::size_t other = 0; other < columns; ++other) {
                if (other == column || !has_direction[other]) {
                    continue;
                }
                const Vector<T> existing_column = u.column(other);
                candidate -= existing_column.dot(candidate) * existing_column;
            }

            const T candidate_norm = candidate.norm_l2();
            if (candidate_norm > best_candidate_norm) {
                best_candidate_norm = candidate_norm;
                best_candidate = std::move(candidate);
            }
        }

        for (std::size_t row = 0; row < rows; ++row) {
            u(row, column) = best_candidate[row] / best_candidate_norm;
        }
        has_direction[column] = true;
    }
}

template <std::floating_point T>
void sort_singular_triplets(Matrix<T>& u, Vector<T>& s, Matrix<T>& v) {
    const std::size_t columns = s.size();

    std::vector<std::size_t> order(columns);
    std::ranges::iota(order, std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&s](std::size_t lhs, std::size_t rhs) { return s[lhs] > s[rhs]; });

    Matrix<T> sorted_u(u.rows(), columns);
    Vector<T> sorted_s(columns);
    Matrix<T> sorted_v(v.rows(), columns);

    for (std::size_t k = 0; k < columns; ++k) {
        const std::size_t source = order[k];
        for (std::size_t i = 0; i < u.rows(); ++i) {
            sorted_u(i, k) = u(i, source);
        }
        sorted_s[k] = s[source];
        for (std::size_t i = 0; i < v.rows(); ++i) {
            sorted_v(i, k) = v(i, source);
        }
    }

    u = std::move(sorted_u);
    s = std::move(sorted_s);
    v = std::move(sorted_v);
}

template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void orient_columns(Matrix<T>& u, Matrix<T>& v) {
    for (std::size_t column = 0; column < u.cols(); ++column) {
        std::size_t largest = 0;
        for (std::size_t i = 1; i < u.rows(); ++i) {
            if (std::abs(u(i, column)) > std::abs(u(largest, column))) {
                largest = i;
            }
        }

        if (u(largest, column) < T{}) {
            for (std::size_t i = 0; i < u.rows(); ++i) {
                u(i, column) = -u(i, column);
            }
            for (std::size_t i = 0; i < v.rows(); ++i) {
                v(i, column) = -v(i, column);
            }
        }
    }
}

} // namespace detail::svd

/// @return The decomposition, or an error if @p matrix contains a non-finite value or the iteration
/// does not converge.
/// @throws std::invalid_argument If @p matrix is empty or @p tolerance is not finite or negative.
template <std::floating_point T>
[[nodiscard]] std::expected<SvdDecomposition<T>, Error>
svd_decompose(const Matrix<T>& matrix,
              T tolerance = static_cast<T>(detail::svd::tolerance_epsilon_multiplier) *
                            std::numeric_limits<T>::epsilon(),
              std::size_t max_sweeps = detail::svd::default_max_sweeps) {
    if (!std::isfinite(tolerance) || tolerance < T{}) {
        throw std::invalid_argument(
            std::format("tolerance must be finite and non-negative: got {}", tolerance));
    }
    detail::require_non_empty_matrix(matrix);
    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    const bool is_wide = matrix.rows() < matrix.cols();
    Matrix<T> work = is_wide ? matrix.transpose() : matrix;
    const std::size_t rows = work.rows();
    const std::size_t columns = work.cols();

    const T scale = detail::svd::max_abs(work);
    if (scale > T{}) {
        work /= scale;
    }

    Matrix<T> v = Matrix<T>::identity(columns);
    std::size_t sweeps = 0;
    while (!detail::svd::is_converged(work, tolerance)) {
        if (sweeps >= max_sweeps) {
            return std::unexpected(Error::no_convergence);
        }

        for (std::size_t p = 0; p < columns; ++p) {
            for (std::size_t q = p + 1; q < columns; ++q) {
                const detail::svd::ColumnGram<T> gram = detail::svd::column_gram(work, p, q);
                if (std::abs(gram.gamma) <= tolerance * std::sqrt(gram.alpha * gram.beta)) {
                    continue;
                }

                const auto rotation =
                    detail::jacobi::make_rotation(gram.alpha, gram.beta, gram.gamma);
                detail::jacobi::rotate_columns(work, p, q, rotation);
                detail::jacobi::rotate_columns(v, p, q, rotation);
            }
        }
        ++sweeps;
    }

    Vector<T> s(columns);
    Matrix<T> u(rows, columns);
    for (std::size_t j = 0; j < columns; ++j) {
        const T norm = detail::svd::column_norm(work, j);
        s[j] = scale * norm;
        if (norm > T{}) {
            for (std::size_t i = 0; i < rows; ++i) {
                u(i, j) = work(i, j) / norm;
            }
        }
    }
    detail::svd::fill_orthonormal_columns(u);

    detail::svd::sort_singular_triplets(u, s, v);
    if (is_wide) {
        std::swap(u, v);
    }
    detail::svd::orient_columns(u, v);

    return SvdDecomposition<T>{std::move(u), std::move(s), std::move(v)};
}

/// @}

} // namespace num::linalg
