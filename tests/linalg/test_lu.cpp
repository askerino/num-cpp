#include "num/linalg/lu.hpp"
#include "support/comparison.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
// Returns a matrix that requires row swaps.
num::Matrix<double> pivot_matrix() { return {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}}; }
} // namespace

TEST(LuDecompose, SatisfiesPAEqualsLU) {
    const auto matrix = pivot_matrix();
    const auto decomposition = num::linalg::lu_decompose(matrix).value();
    ASSERT_EQ(decomposition.permutation.size(), matrix.rows());

    num::Matrix<double> permuted(matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            permuted(row, column) = matrix(decomposition.permutation[row], column);
        }
    }
    num::test::expect_matrix_near(decomposition.lower * decomposition.upper, permuted);
}

TEST(LuDecompose, ReturnsTriangularFactors) {
    const auto matrix = pivot_matrix();
    const auto decomposition = num::linalg::lu_decompose(matrix).value();
    const std::size_t n = matrix.rows();

    for (std::size_t row = 0; row < n; ++row) {
        EXPECT_DOUBLE_EQ(decomposition.lower(row, row), 1.0);
        for (std::size_t column = row + 1; column < n; ++column) {
            EXPECT_DOUBLE_EQ(decomposition.lower(row, column), 0.0);
        }
        for (std::size_t column = 0; column < row; ++column) {
            EXPECT_DOUBLE_EQ(decomposition.upper(row, column), 0.0);
        }
    }
}

TEST(LuDecompose, CountsRowSwaps) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_decomposition = num::linalg::lu_decompose(no_swap).value();
    const auto one_swap_decomposition = num::linalg::lu_decompose(one_swap).value();

    EXPECT_EQ(no_swap_decomposition.row_swaps, 0);
    EXPECT_EQ(one_swap_decomposition.row_swaps, 1);
}

TEST(LuDecompose, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto decomposition = num::linalg::lu_decompose(non_finite);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_finite_input);
}

TEST(LuDecompose, ReportsSingularMatrix) {
    const num::Matrix<double> singular{{1.0, 2.0}, {2.0, 4.0}};
    const num::Matrix<double> zero{{0.0}};
    const auto singular_decomposition = num::linalg::lu_decompose(singular);
    const auto zero_decomposition = num::linalg::lu_decompose(zero);

    ASSERT_FALSE(singular_decomposition.has_value());
    EXPECT_EQ(singular_decomposition.error(), num::Error::singular_matrix);
    ASSERT_FALSE(zero_decomposition.has_value());
    EXPECT_EQ(zero_decomposition.error(), num::Error::singular_matrix);
}

TEST(LuDecompose, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::lu_decompose(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::lu_decompose(non_square)), std::invalid_argument);
}

TEST(LuDeterminant, Computes) {
    const num::Matrix<double> no_swap{{2.0, 1.0}, {0.0, 3.0}};
    const num::Matrix<double> one_swap{{0.0, 1.0}, {1.0, 0.0}};
    const auto no_swap_decomposition = num::linalg::lu_decompose(no_swap).value();
    const auto one_swap_decomposition = num::linalg::lu_decompose(one_swap).value();

    EXPECT_DOUBLE_EQ(num::linalg::lu_determinant(no_swap_decomposition), 6.0);
    EXPECT_DOUBLE_EQ(num::linalg::lu_determinant(one_swap_decomposition), -1.0);
}

TEST(LuSolve, Solves) {
    const num::Matrix<double> matrix{{3.0, 2.0, -1.0}, {2.0, -2.0, 4.0}, {-1.0, 0.5, -1.0}};
    const num::Vector<double> rhs{1.0, -2.0, 0.0};
    const auto decomposition = num::linalg::lu_decompose(matrix).value();
    const auto solution = num::linalg::lu_solve(decomposition, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{1.0, -2.0, -2.0});
}

TEST(LuSolve, ReportsNonFiniteRhsInput) {
    const auto decomposition = num::linalg::lu_decompose(num::Matrix<double>::identity(2)).value();
    const num::Vector<double> non_finite_rhs{1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto solution = num::linalg::lu_solve(decomposition, non_finite_rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(LuSolve, ThrowsOnDimensionMismatch) {
    const auto matrix = num::Matrix<double>::identity(2);
    const auto decomposition = num::linalg::lu_decompose(matrix).value();
    const num::Vector<double> rhs{1.0};

    EXPECT_THROW(static_cast<void>(num::linalg::lu_solve(decomposition, rhs)),
                 std::invalid_argument);
}

TEST(LuInverse, Computes) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {1.0, 3.0}};
    const auto decomposition = num::linalg::lu_decompose(matrix).value();
    const auto inverse = num::linalg::lu_inverse(decomposition);
    const auto matrix_times_inverse = matrix * inverse;

    num::test::expect_matrix_near(matrix_times_inverse,
                                  num::Matrix<double>::identity(matrix.rows()));
}
