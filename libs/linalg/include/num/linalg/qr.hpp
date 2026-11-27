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
#include <utility>

namespace num::linalg {

/// @defgroup qr QR decomposition
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct QrDecomposition {
    Matrix<T> q;
    /// Upper triangular factor. Diagonal entries may be negative.
    Matrix<T> r;
};

namespace detail::qr {

// Assumes r_kk is already sign-adjusted to avoid cancellation.
template <std::floating_point T>
[[nodiscard]] Vector<T> householder_vector(const Matrix<T>& a, std::size_t k, T r_kk) {
    Vector<T> v(a.rows());
    for (std::size_t i = k; i < a.rows(); ++i) {
        v[i] = a(i, k);
    }
    v[k] -= r_kk;
    v /= v.norm_l2();
    return v;
}

// H = I - 2 v v^T
template <std::floating_point T>
[[nodiscard]] Matrix<T> householder_matrix(const Vector<T>& v) {
    const std::size_t m = v.size();
    Matrix<T> h = Matrix<T>::identity(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            h(i, j) -= T{2} * v[i] * v[j];
        }
    }
    return h;
}

} // namespace detail::qr

/// @note @p matrix is reported as rank deficient only when r_kk is exactly zero.
/// @return The decomposition, or an error if @p matrix is rank deficient or contains a non-finite
/// value.
/// @throws std::invalid_argument If @p matrix is empty or has fewer rows than columns.
template <std::floating_point T>
[[nodiscard]] std::expected<QrDecomposition<T>, Error> qr_decompose(const Matrix<T>& matrix) {
    detail::require_tall_or_square_matrix(matrix);
    if (!detail::all_finite(matrix)) {
        return std::unexpected(Error::non_finite_input);
    }

    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    Matrix<T> a = matrix;
    Matrix<T> q = Matrix<T>::identity(m);

    for (std::size_t k = 0; k < n; ++k) {
        const T a_kk = a(k, k);
        T sigma{};
        for (std::size_t i = k + 1; i < m; ++i) {
            sigma = std::hypot(sigma, a(i, k));
        }

        if (sigma == T{}) {
            if (a_kk == T{}) {
                return std::unexpected(Error::rank_deficient);
            }
            continue;
        }

        // The sign opposite to a_kk avoids cancellation in v_k = a_kk - r_kk.
        const T r_kk = -std::copysign(std::hypot(a_kk, sigma), a_kk);
        const Vector<T> v = detail::qr::householder_vector(a, k, r_kk);
        const Matrix<T> h = detail::qr::householder_matrix(v);
        q = q * h;
        a = h * a;
    }

    // Copy upper triangle only; entries of a below the diagonal are rounding errors.
    Matrix<T> thin_q(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            thin_q(i, j) = q(i, j);
        }
    }
    Matrix<T> r(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            r(i, j) = a(i, j);
        }
    }

    return QrDecomposition<T>{std::move(thin_q), std::move(r)};
}

/// @return The least-squares solution, or an error if @p rhs contains a non-finite value.
/// @throws std::invalid_argument If @p rhs has an incompatible size.
template <std::floating_point T>
[[nodiscard]] std::expected<Vector<T>, Error>
qr_least_squares(const QrDecomposition<T>& decomposition, const Vector<T>& rhs) {
    detail::require_rhs_size(decomposition.q.rows(), rhs);
    if (!detail::all_finite(rhs)) {
        return std::unexpected(Error::non_finite_input);
    }

    const Vector<T> projected_rhs = decomposition.q.transpose() * rhs;
    return detail::backward_substitution(decomposition.r, projected_rhs);
}

/// @}

} // namespace num::linalg
