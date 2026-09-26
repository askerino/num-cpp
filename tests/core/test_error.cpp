#include "num/core/error.hpp"

#include <gtest/gtest.h>

TEST(ErrorMessage, ReturnsMessages) {
    EXPECT_EQ(num::error_message(num::Error::singular_matrix), "matrix is singular");
    EXPECT_EQ(num::error_message(num::Error::non_finite_input),
              "input contains a non-finite value");
}

TEST(ErrorMessage, ReturnsFallbackForUnknownValue) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    EXPECT_EQ(num::error_message(static_cast<num::Error>(255)), "unknown error");
}
