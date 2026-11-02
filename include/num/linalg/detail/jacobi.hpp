#pragma once

#include "num/core/matrix.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>

namespace num::linalg::detail::jacobi {

template <std::floating_point T>
struct Rotation {
    T cos;
    T sin;
    T tan;
};

/// @pre @p gamma must not be zero.
template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] Rotation<T> make_rotation(T alpha, T beta, T gamma) {
    const T tau = (beta - alpha) / (T{2} * gamma);
    const T tangent =
        tau == T{} ? T{1} : std::copysign(T{1}, tau) / (std::abs(tau) + std::hypot(T{1}, tau));
    const T cosine = T{1} / std::hypot(T{1}, tangent);
    return Rotation<T>{cosine, tangent * cosine, tangent};
}

template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void rotate_columns(Matrix<T>& matrix, std::size_t p, std::size_t q, const Rotation<T>& rotation) {
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        const T x = matrix(i, p);
        const T y = matrix(i, q);
        matrix(i, p) = (rotation.cos * x) - (rotation.sin * y);
        matrix(i, q) = (rotation.sin * x) + (rotation.cos * y);
    }
}

} // namespace num::linalg::detail::jacobi
