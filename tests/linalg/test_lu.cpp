#include "num/linalg/lu.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
constexpr double double_tolerance = 1e-12;
} // namespace

TEST(LuFactorize, SatisfiesPAEqualsLU) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
    const auto factorization = num::linalg::lu_factorize(matrix).value();
    const auto lu_product = factorization.lower * factorization.upper;

    ASSERT_EQ(factorization.permutation.size(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            EXPECT_NEAR(lu_product(row, column), matrix(factorization.permutation[row], column),
                        double_tolerance);
        }
    }
}

TEST(LuFactorize, ReturnsTriangularFactors) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
    const auto factorization = num::linalg::lu_factorize(matrix).value();

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        EXPECT_DOUBLE_EQ(factorization.lower(row, row), 1.0);
        for (std::size_t column = row + 1; column < matrix.cols(); ++column) {
            EXPECT_DOUBLE_EQ(factorization.lower(row, column), 0.0);
            const std::size_t upper_row = column;
            const std::size_t upper_column = row;
            EXPECT_DOUBLE_EQ(factorization.upper(upper_row, upper_column), 0.0);
        }
    }
}

TEST(LuFactorize, CountsRowSwaps) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_factorization = num::linalg::lu_factorize(no_swap).value();
    const auto one_swap_factorization = num::linalg::lu_factorize(one_swap).value();

    EXPECT_EQ(no_swap_factorization.row_swaps, 0);
    EXPECT_EQ(one_swap_factorization.row_swaps, 1);
}

TEST(LuFactorize, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto factorization = num::linalg::lu_factorize(non_finite);

    ASSERT_FALSE(factorization.has_value());
    EXPECT_EQ(factorization.error(), num::Error::non_finite_input);
}

TEST(LuFactorize, ReportsSingularMatrix) {
    const num::Matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const num::Matrix<double> zero{{0.0}};
    const auto singular_factorization = num::linalg::lu_factorize(singular);
    const auto zero_factorization = num::linalg::lu_factorize(zero);

    ASSERT_FALSE(singular_factorization.has_value());
    EXPECT_EQ(singular_factorization.error(), num::Error::singular_matrix);
    ASSERT_FALSE(zero_factorization.has_value());
    EXPECT_EQ(zero_factorization.error(), num::Error::singular_matrix);
}

TEST(LuFactorize, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW([[maybe_unused]] const auto result = num::linalg::lu_factorize(empty),
                 std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::linalg::lu_factorize(non_square),
                 std::invalid_argument);
}

TEST(LuDeterminant, Computes) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_factorization = num::linalg::lu_factorize(no_swap).value();
    const auto one_swap_factorization = num::linalg::lu_factorize(one_swap).value();

    EXPECT_DOUBLE_EQ(num::linalg::lu_determinant(no_swap_factorization), 6.0);
    EXPECT_DOUBLE_EQ(num::linalg::lu_determinant(one_swap_factorization), -1.0);
}

TEST(LuSolve, Solves) {
    const num::Matrix<double> matrix{{3.0, 2.0, -1.0}, {2.0, -2.0, 4.0}, {-1.0, 0.5, -1.0}};
    const num::Vector<double> rhs{1.0, -2.0, 0.0};
    const auto factorization = num::linalg::lu_factorize(matrix).value();
    const auto solution = num::linalg::lu_solve(factorization, rhs).value();

    EXPECT_NEAR(solution[0], 1.0, double_tolerance);
    EXPECT_NEAR(solution[1], -2.0, double_tolerance);
    EXPECT_NEAR(solution[2], -2.0, double_tolerance);
}

TEST(LuSolve, ReportsNonFiniteRhsInput) {
    const auto factorization = num::linalg::lu_factorize(num::Matrix<double>::identity(2)).value();
    const num::Vector<double> non_finite_rhs{1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto solution = num::linalg::lu_solve(factorization, non_finite_rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(LuSolve, ThrowsOnDimensionMismatch) {
    const auto matrix = num::Matrix<double>::identity(2);
    const auto factorization = num::linalg::lu_factorize(matrix).value();
    const num::Vector<double> rhs{1.0};

    EXPECT_THROW([[maybe_unused]] const auto result = num::linalg::lu_solve(factorization, rhs),
                 std::invalid_argument);
}

TEST(LuInverse, Computes) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {1.0, 3.0}};
    const auto factorization = num::linalg::lu_factorize(matrix).value();
    const auto inverse = num::linalg::lu_inverse(factorization);
    const auto matrix_times_inverse = matrix * inverse;

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            const double expected = (row == column) ? 1.0 : 0.0;
            EXPECT_NEAR(matrix_times_inverse(row, column), expected, double_tolerance);
        }
    }
}
