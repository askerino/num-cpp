#include "num/linalg/error.hpp"

#include <gtest/gtest.h>

TEST(LinalgErrorTest, ReturnsMessages) {
    EXPECT_EQ(num::linalg_error_message(num::LinalgError::singular_matrix), "matrix is singular");
    EXPECT_EQ(num::linalg_error_message(num::LinalgError::non_finite_input),
              "input contains a non-finite value");
}

TEST(LinalgErrorTest, ReturnsFallbackForUnknownValue) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    EXPECT_EQ(num::linalg_error_message(static_cast<num::LinalgError>(255)),
              "unknown linear algebra error");
}
