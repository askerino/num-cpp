#pragma once

#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"

#include <cstddef>
#include <gtest/gtest.h>

namespace num::test {

inline constexpr double tolerance = 1e-12;

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

} // namespace num::test
