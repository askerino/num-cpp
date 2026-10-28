#pragma once

#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <ranges>
#include <stdexcept>

namespace num::linalg::detail {

template <typename Range>
    requires std::floating_point<std::ranges::range_value_t<Range>>
[[nodiscard]] bool all_finite(const Range& values) {
    return std::ranges::all_of(values, [](const auto value) { return std::isfinite(value); });
}

template <std::floating_point T>
void require_square_matrix(const Matrix<T>& matrix) {
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument(std::format("matrix must be non-empty and square: got {}x{}",
                                                matrix.rows(), matrix.cols()));
    }
}

template <std::floating_point T>
void require_tall_or_square_matrix(const Matrix<T>& matrix) {
    if (matrix.rows() == 0 || matrix.cols() == 0 || matrix.rows() < matrix.cols()) {
        throw std::invalid_argument(std::format(
            "matrix must be non-empty and have at least as many rows as columns: got {}x{}",
            matrix.rows(), matrix.cols()));
    }
}

template <std::floating_point T>
void require_rhs_size(std::size_t system_size, const Vector<T>& rhs) {
    if (rhs.size() != system_size) {
        throw std::invalid_argument(std::format("solve size mismatch: system size {}, rhs size {}",
                                                system_size, rhs.size()));
    }
}

} // namespace num::linalg::detail
