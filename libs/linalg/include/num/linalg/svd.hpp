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
#include <functional>
#include <limits>
#include <numeric>
#include <ranges>
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
[[nodiscard]] T rank_threshold(const SvdDecomposition<T>& decomposition) {
    const std::size_t max_dimension = std::max(decomposition.u.rows(), decomposition.v.rows());
    return static_cast<T>(max_dimension) * std::numeric_limits<T>::epsilon() * decomposition.s[0];
}

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
        const T a_ip = matrix(i, p);
        const T a_iq = matrix(i, q);
        gram.alpha += a_ip * a_ip;
        gram.beta += a_iq * a_iq;
        gram.gamma += a_ip * a_iq;
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
void fill_orthonormal_columns(Matrix<T>& u) {
    const std::size_t m = u.rows();
    const std::size_t n = u.cols();

    std::vector<bool> has_direction(n, false);
    for (std::size_t j = 0; j < n; ++j) {
        has_direction[j] = u.column(j).norm_l2() > T{};
    }

    for (std::size_t j = 0; j < n; ++j) {
        if (has_direction[j]) {
            continue;
        }

        Vector<T> best_candidate(m);
        T best_candidate_norm{};
        for (std::size_t basis_index = 0; basis_index < m; ++basis_index) {
            Vector<T> candidate(m);
            candidate[basis_index] = T{1};
            for (std::size_t k = 0; k < n; ++k) {
                if (k == j || !has_direction[k]) {
                    continue;
                }
                const Vector<T> existing_column = u.column(k);
                candidate -= existing_column.dot(candidate) * existing_column;
            }

            const T candidate_norm = candidate.norm_l2();
            if (candidate_norm > best_candidate_norm) {
                best_candidate_norm = candidate_norm;
                best_candidate = std::move(candidate);
            }
        }

        for (std::size_t i = 0; i < m; ++i) {
            u(i, j) = best_candidate[i] / best_candidate_norm;
        }
        has_direction[j] = true;
    }
}

template <std::floating_point T>
void sort_singular_triplets(Matrix<T>& u, Vector<T>& s, Matrix<T>& v) {
    const std::size_t n = s.size();

    std::vector<std::size_t> order(n);
    std::ranges::iota(order, std::size_t{0});
    std::ranges::stable_sort(order, std::ranges::greater{}, [&s](std::size_t k) { return s[k]; });

    Matrix<T> sorted_u(u.rows(), n);
    Vector<T> sorted_s(n);
    Matrix<T> sorted_v(v.rows(), n);

    for (std::size_t k = 0; k < n; ++k) {
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
    for (std::size_t k = 0; k < u.cols(); ++k) {
        const auto magnitude = [&u, k](std::size_t i) { return std::abs(u(i, k)); };
        const std::size_t largest =
            *std::ranges::max_element(std::views::iota(std::size_t{0}, u.rows()), {}, magnitude);

        if (u(largest, k) < T{}) {
            for (std::size_t i = 0; i < u.rows(); ++i) {
                u(i, k) = -u(i, k);
            }
            for (std::size_t i = 0; i < v.rows(); ++i) {
                v(i, k) = -v(i, k);
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
    detail::require_valid_tolerance(tolerance);
    detail::require_non_empty_matrix(matrix);
    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    const bool is_wide = matrix.rows() < matrix.cols();
    Matrix<T> a = is_wide ? matrix.transpose() : matrix;
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();

    const T scale = detail::svd::max_abs(a);
    if (scale > T{}) {
        a /= scale;
    }

    Matrix<T> v = Matrix<T>::identity(n);
    std::size_t sweeps = 0;
    while (!detail::svd::is_converged(a, tolerance)) {
        if (sweeps >= max_sweeps) {
            return std::unexpected(Error::no_convergence);
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const detail::svd::ColumnGram<T> gram = detail::svd::column_gram(a, p, q);
                if (std::abs(gram.gamma) <= tolerance * std::sqrt(gram.alpha * gram.beta)) {
                    continue;
                }

                const auto rotation =
                    detail::jacobi::make_rotation(gram.alpha, gram.beta, gram.gamma);
                detail::jacobi::rotate_columns(a, p, q, rotation);
                detail::jacobi::rotate_columns(v, p, q, rotation);
            }
        }
        ++sweeps;
    }

    Vector<T> s(n);
    Matrix<T> u(m, n);
    for (std::size_t j = 0; j < n; ++j) {
        const T norm = a.column(j).norm_l2();
        s[j] = scale * norm;
        if (norm > T{}) {
            for (std::size_t i = 0; i < m; ++i) {
                u(i, j) = a(i, j) / norm;
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

template <std::floating_point T>
[[nodiscard]] std::size_t svd_rank(const SvdDecomposition<T>& decomposition) {
    const T threshold = detail::svd::rank_threshold(decomposition);
    return static_cast<std::size_t>(
        std::ranges::count_if(decomposition.s, [threshold](T value) { return value > threshold; }));
}

/// @return The least-squares solution, or an error if @p rhs contains a non-finite
/// value.
/// @throws std::invalid_argument If @p rhs has an incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, Error>
svd_least_squares(const SvdDecomposition<T>& decomposition, const Vector<T>& rhs) {
    detail::require_rhs_size(decomposition.u.rows(), rhs);
    if (!detail::all_finite(rhs)) {
        return std::unexpected(Error::non_finite_input);
    }

    const Vector<T> projected_rhs = decomposition.u.transpose() * rhs;
    const T threshold = detail::svd::rank_threshold(decomposition);
    Vector<T> solution(decomposition.v.rows());

    for (std::size_t k = 0; k < decomposition.s.size(); ++k) {
        if (decomposition.s[k] <= threshold) {
            continue;
        }
        solution += (projected_rhs[k] / decomposition.s[k]) * decomposition.v.column(k);
    }

    return solution;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> svd_pseudo_inverse(const SvdDecomposition<T>& decomposition) {
    const std::size_t m = decomposition.u.rows();
    const std::size_t n = decomposition.v.rows();
    const T threshold = detail::svd::rank_threshold(decomposition);
    Matrix<T> inverse(n, m);

    for (std::size_t k = 0; k < decomposition.s.size(); ++k) {
        if (decomposition.s[k] <= threshold) {
            continue;
        }
        const T inverse_singular_value = T{1} / decomposition.s[k];
        for (std::size_t i = 0; i < n; ++i) {
            const T v_scaled = decomposition.v(i, k) * inverse_singular_value;
            for (std::size_t j = 0; j < m; ++j) {
                inverse(i, j) += v_scaled * decomposition.u(j, k);
            }
        }
    }

    return inverse;
}

template <std::floating_point T>
[[nodiscard]] T svd_condition_number(const SvdDecomposition<T>& decomposition) {
    const T s_min = decomposition.s[decomposition.s.size() - 1];
    if (s_min <= detail::svd::rank_threshold(decomposition)) {
        return std::numeric_limits<T>::infinity();
    }
    return decomposition.s[0] / s_min;
}

/// @}

} // namespace num::linalg
