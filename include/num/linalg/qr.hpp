#pragma once

#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
#include "num/linalg/detail/triangular.hpp"
#include "num/linalg/detail/validation.hpp"

#include <concepts>
#include <cstddef>
#include <expected>

namespace num::linalg {

/// @defgroup qr QR decomposition
/// @ingroup linalg
/// @{

template <std::floating_point T>
struct QrDecomposition {
    Matrix<T> q;
    Matrix<T> r;
};

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
    Matrix<T> q(m, n);
    Matrix<T> r(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        Vector<T> v = matrix.column(j);

        for (std::size_t k = 0; k < j; ++k) {
            const auto q_k = q.column(k);
            const T r_kj = q_k.dot(v);
            r(k, j) = r_kj;
            v -= r_kj * q_k;
        }

        const T r_jj = v.norm_l2();
        if (r_jj == T{}) {
            return std::unexpected(Error::rank_deficient);
        }
        r(j, j) = r_jj;

        for (std::size_t i = 0; i < m; ++i) {
            q(i, j) = v[i] / r_jj;
        }
    }

    return QrDecomposition<T>{q, r};
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
