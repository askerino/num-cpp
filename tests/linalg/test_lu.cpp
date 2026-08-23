#include "num/linalg/lu.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
constexpr double double_tolerance = 1e-12;
} // namespace

TEST(LuTest, SatisfiesPAEqualsLU) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
    const auto factorization = num::lu_factorize(matrix).value();
    const auto lu_product = factorization.lower * factorization.upper;

    ASSERT_EQ(factorization.permutation.size(), matrix.rows());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            EXPECT_NEAR(lu_product(row, column), matrix(factorization.permutation[row], column),
                        double_tolerance);
        }
    }
}

TEST(LuTest, ReturnsTriangularFactors) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
    const auto factorization = num::lu_factorize(matrix).value();

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

TEST(LuTest, CountsRowSwaps) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_factorization = num::lu_factorize(no_swap).value();
    const auto one_swap_factorization = num::lu_factorize(one_swap).value();

    EXPECT_EQ(no_swap_factorization.row_swaps, 0);
    EXPECT_EQ(one_swap_factorization.row_swaps, 1);
}

TEST(LuTest, RejectsInvalidFactorizationInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_factorize(empty),
                 std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_factorize(non_square),
                 std::invalid_argument);
}

TEST(LuTest, ReportsNonFiniteFactorizationInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto factorization = num::lu_factorize(non_finite);

    ASSERT_FALSE(factorization.has_value());
    EXPECT_EQ(factorization.error(), num::LinalgError::non_finite_input);
}

TEST(LuTest, ReportsSingularFactorization) {
    const num::Matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const num::Matrix<double> zero{{0.0}};
    const auto singular_factorization = num::lu_factorize(singular);
    const auto zero_factorization = num::lu_factorize(zero);

    ASSERT_FALSE(singular_factorization.has_value());
    EXPECT_EQ(singular_factorization.error(), num::LinalgError::singular_matrix);
    ASSERT_FALSE(zero_factorization.has_value());
    EXPECT_EQ(zero_factorization.error(), num::LinalgError::singular_matrix);
}

TEST(LuTest, ComputesDeterminants) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_factorization = num::lu_factorize(no_swap).value();
    const auto one_swap_factorization = num::lu_factorize(one_swap).value();

    EXPECT_DOUBLE_EQ(num::determinant(no_swap).value(), 6.0);
    EXPECT_DOUBLE_EQ(num::determinant(one_swap).value(), -1.0);
    EXPECT_DOUBLE_EQ(num::determinant(no_swap_factorization), 6.0);
    EXPECT_DOUBLE_EQ(num::determinant(one_swap_factorization), -1.0);
}

TEST(LuTest, RejectsInvalidDeterminantInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW([[maybe_unused]] const auto result = num::determinant(empty),
                 std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::determinant(non_square),
                 std::invalid_argument);
}

TEST(LuTest, ReportsNonFiniteDeterminantInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto determinant = num::determinant(non_finite);

    ASSERT_FALSE(determinant.has_value());
    EXPECT_EQ(determinant.error(), num::LinalgError::non_finite_input);
}

TEST(LuTest, ReturnsZeroForSingularDeterminant) {
    const num::Matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};

    EXPECT_DOUBLE_EQ(num::determinant(singular).value(), 0.0);
}

TEST(LuTest, SolvesFromFactorization) {
    const num::Matrix<double> matrix{{2.0, 3.0}, {4.0, 5.0}};
    const num::Vector<double> rhs{8.0, 14.0};
    const auto factorization = num::lu_factorize(matrix).value();
    const auto solution = num::lu_solve(factorization, rhs).value();

    EXPECT_NEAR(solution[0], 1.0, double_tolerance);
    EXPECT_NEAR(solution[1], 2.0, double_tolerance);
}

TEST(LuTest, SolvesFromMatrix) {
    const num::Matrix<double> matrix{{3.0, 2.0, -1.0}, {2.0, -2.0, 4.0}, {-1.0, 0.5, -1.0}};
    const num::Vector<double> rhs{1.0, -2.0, 0.0};
    const auto solution = num::lu_solve(matrix, rhs).value();

    EXPECT_NEAR(solution[0], 1.0, double_tolerance);
    EXPECT_NEAR(solution[1], -2.0, double_tolerance);
    EXPECT_NEAR(solution[2], -2.0, double_tolerance);
}

TEST(LuTest, RejectsSolveDimensionMismatch) {
    const auto matrix = num::Matrix<double>::identity(2);
    const auto factorization = num::lu_factorize(matrix).value();
    const num::Vector<double> rhs{1.0};

    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_solve(factorization, rhs),
                 std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_solve(matrix, rhs),
                 std::invalid_argument);
}

TEST(LuTest, RejectsInvalidSolveInputs) {
    const num::Matrix<double> non_square_matrix(2, 3);
    const num::Matrix<double> empty_matrix;
    const num::Vector<double> rhs{1.0, 2.0};
    const num::Vector<double> empty_rhs;

    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_solve(non_square_matrix, rhs),
                 std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::lu_solve(empty_matrix, empty_rhs),
                 std::invalid_argument);
}

TEST(LuTest, ReportsNonFiniteMatrixInputForSolve) {
    const num::Matrix<double> non_finite_matrix{{1.0, std::numeric_limits<double>::infinity()},
                                                {0.0, 1.0}};
    const num::Vector<double> rhs{1.0, 2.0};
    const auto solution = num::lu_solve(non_finite_matrix, rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::LinalgError::non_finite_input);
}

TEST(LuTest, ReportsNonFiniteRhsForSolve) {
    const auto matrix = num::Matrix<double>::identity(2);
    const auto factorization = num::lu_factorize(matrix).value();
    const num::Vector<double> non_finite_rhs{1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto solution_from_factorization = num::lu_solve(factorization, non_finite_rhs);
    const auto solution_from_matrix = num::lu_solve(matrix, non_finite_rhs);

    ASSERT_FALSE(solution_from_factorization.has_value());
    EXPECT_EQ(solution_from_factorization.error(), num::LinalgError::non_finite_input);
    ASSERT_FALSE(solution_from_matrix.has_value());
    EXPECT_EQ(solution_from_matrix.error(), num::LinalgError::non_finite_input);
}

TEST(LuTest, ReportsSingularSolve) {
    const num::Matrix<double> singular_matrix{{1.0, 2.0}, {2.0, 4.0}};
    const num::Vector<double> rhs{1.0, 2.0};
    const auto solution = num::lu_solve(singular_matrix, rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::LinalgError::singular_matrix);
}

TEST(LuTest, ComputesInverseFromFactorization) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {1.0, 3.0}};
    const auto inverse = num::inverse(num::lu_factorize(matrix).value());
    const auto matrix_times_inverse = matrix * inverse;

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            const double expected = (row == column) ? 1.0 : 0.0;
            EXPECT_NEAR(matrix_times_inverse(row, column), expected, double_tolerance);
        }
    }
}

TEST(LuTest, ComputesInverseFromMatrix) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {1.0, 3.0}};
    const auto inverse = num::inverse(matrix).value();
    const auto matrix_times_inverse = matrix * inverse;

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            const double expected = (row == column) ? 1.0 : 0.0;
            EXPECT_NEAR(matrix_times_inverse(row, column), expected, double_tolerance);
        }
    }
}

TEST(LuTest, RejectsInvalidInverseInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW([[maybe_unused]] const auto result = num::inverse(empty), std::invalid_argument);
    EXPECT_THROW([[maybe_unused]] const auto result = num::inverse(non_square),
                 std::invalid_argument);
}

TEST(LuTest, ReportsNonFiniteInverseInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto inverse = num::inverse(non_finite);

    ASSERT_FALSE(inverse.has_value());
    EXPECT_EQ(inverse.error(), num::LinalgError::non_finite_input);
}

TEST(LuTest, ReportsSingularInverse) {
    const num::Matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const auto inverse = num::inverse(singular);

    ASSERT_FALSE(inverse.has_value());
    EXPECT_EQ(inverse.error(), num::LinalgError::singular_matrix);
}
