#pragma once

#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"

#include <concepts>
#include <cstddef>

namespace num::linalg::detail {

template <std::floating_point T>
[[nodiscard]] Vector<T> forward_substitution(const Matrix<T>& lower, const Vector<T>& rhs) {
    const std::size_t n = lower.rows();
    Vector<T> y(n);

    for (std::size_t i = 0; i < n; ++i) {
        T sum{};
        for (std::size_t j = 0; j < i; ++j) {
            sum += lower(i, j) * y[j];
        }
        y[i] = (rhs[i] - sum) / lower(i, i);
    }

    return y;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> backward_substitution(const Matrix<T>& upper, const Vector<T>& rhs) {
    const std::size_t n = upper.rows();
    Vector<T> x(n);

    for (std::size_t i = n; i > 0;) {
        --i;
        T sum{};
        for (std::size_t j = i + 1; j < n; ++j) {
            sum += upper(i, j) * x[j];
        }
        x[i] = (rhs[i] - sum) / upper(i, i);
    }

    return x;
}

} // namespace num::linalg::detail
