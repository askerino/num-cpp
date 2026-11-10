#pragma once

#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <ranges>

namespace num::test {

inline constexpr double tolerance = 1e-12;
inline constexpr float float_tolerance = 1e-4F;

inline void expect_vector_near(const Vector<double>& actual, const Vector<double>& expected,
                               double absolute_tolerance = tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], absolute_tolerance) << "at index " << i;
    }
}

inline void expect_matrix_near(const Matrix<double>& actual, const Matrix<double>& expected,
                               double absolute_tolerance = tolerance) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (std::size_t i = 0; i < actual.rows(); ++i) {
        for (std::size_t j = 0; j < actual.cols(); ++j) {
            EXPECT_NEAR(actual(i, j), expected(i, j), absolute_tolerance)
                << "at (" << i << ", " << j << ")";
        }
    }
}

inline void expect_lower_triangular(const Matrix<double>& matrix) {
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = i + 1; j < matrix.cols(); ++j) {
            EXPECT_DOUBLE_EQ(matrix(i, j), 0.0) << "at (" << i << ", " << j << ")";
        }
    }
}

inline void expect_upper_triangular(const Matrix<double>& matrix) {
    for (std::size_t i = 1; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < i && j < matrix.cols(); ++j) {
            EXPECT_DOUBLE_EQ(matrix(i, j), 0.0) << "at (" << i << ", " << j << ")";
        }
    }
}

inline void expect_orthonormal_columns(const Matrix<double>& matrix) {
    expect_matrix_near(matrix.transpose() * matrix, Matrix<double>::identity(matrix.cols()));
}

inline void expect_oriented_columns(const Matrix<double>& matrix) {
    for (std::size_t j = 0; j < matrix.cols(); ++j) {
        const auto magnitude = [&matrix, j](std::size_t i) { return std::abs(matrix(i, j)); };
        const std::size_t largest = *std::ranges::max_element(
            std::views::iota(std::size_t{0}, matrix.rows()), {}, magnitude);
        EXPECT_GE(matrix(largest, j), 0.0) << "in column " << j;
    }
}

} // namespace num::test
