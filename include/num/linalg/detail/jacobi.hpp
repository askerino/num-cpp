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
    const T t =
        tau == T{} ? T{1} : std::copysign(T{1}, tau) / (std::abs(tau) + std::hypot(T{1}, tau));
    const T c = T{1} / std::hypot(T{1}, t);
    return Rotation<T>{c, t * c, t};
}

template <std::floating_point T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void rotate_columns(Matrix<T>& matrix, std::size_t p, std::size_t q, const Rotation<T>& rotation) {
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        const T a_ip = matrix(i, p);
        const T a_iq = matrix(i, q);
        matrix(i, p) = (rotation.cos * a_ip) - (rotation.sin * a_iq);
        matrix(i, q) = (rotation.sin * a_ip) + (rotation.cos * a_iq);
    }
}

} // namespace num::linalg::detail::jacobi
