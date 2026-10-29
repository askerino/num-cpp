#include "num/linalg/cholesky.hpp"
#include "num/linalg/lu.hpp"
#include "support/comparison.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
// Returns a symmetric positive definite matrix.
num::Matrix<double> spd_matrix() {
    return {{4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};
}
} // namespace

TEST(CholeskyDecompose, ComputesLowerFactor) {
    const auto decomposition = num::linalg::cholesky_decompose(spd_matrix()).value();

    num::test::expect_matrix_near(
        decomposition.lower,
        num::Matrix<double>{{2.0, 0.0, 0.0}, {6.0, 1.0, 0.0}, {-8.0, 5.0, 3.0}});
}

TEST(CholeskyDecompose, SatisfiesAEqualsLLt) {
    const auto matrix = spd_matrix();
    const auto decomposition = num::linalg::cholesky_decompose(matrix).value();
    const auto product = decomposition.lower * decomposition.lower.transpose();

    num::test::expect_matrix_near(product, matrix);
}

TEST(CholeskyDecompose, ReturnsLowerTriangular) {
    const auto decomposition = num::linalg::cholesky_decompose(spd_matrix()).value();

    for (std::size_t row = 0; row < decomposition.lower.rows(); ++row) {
        for (std::size_t column = row + 1; column < decomposition.lower.cols(); ++column) {
            EXPECT_DOUBLE_EQ(decomposition.lower(row, column), 0.0);
        }
    }
}

TEST(CholeskyDecompose, UsesOnlyLowerTriangle) {
    const num::Matrix<double> not_symmetric{{4.0, 100.0}, {1.0, 4.0}};
    const auto decomposition = num::linalg::cholesky_decompose(not_symmetric).value();
    const auto product = decomposition.lower * decomposition.lower.transpose();

    num::test::expect_matrix_near(product, num::Matrix<double>{{4.0, 1.0}, {1.0, 4.0}});
}

TEST(CholeskyDecompose, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto decomposition = num::linalg::cholesky_decompose(non_finite);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_finite_input);
}

TEST(CholeskyDecompose, ReportsNonPositiveDefinite) {
    const num::Matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    const num::Matrix<double> negative_diagonal{{-1.0, 0.0}, {0.0, 1.0}};
    const num::Matrix<double> zero{{0.0}};

    const auto indefinite_result = num::linalg::cholesky_decompose(indefinite);
    const auto negative_diagonal_result = num::linalg::cholesky_decompose(negative_diagonal);
    const auto zero_result = num::linalg::cholesky_decompose(zero);

    ASSERT_FALSE(indefinite_result.has_value());
    EXPECT_EQ(indefinite_result.error(), num::Error::non_positive_definite);
    ASSERT_FALSE(negative_diagonal_result.has_value());
    EXPECT_EQ(negative_diagonal_result.error(), num::Error::non_positive_definite);
    ASSERT_FALSE(zero_result.has_value());
    EXPECT_EQ(zero_result.error(), num::Error::non_positive_definite);
}

TEST(CholeskyDecompose, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_decompose(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_decompose(non_square)),
                 std::invalid_argument);
}

TEST(CholeskyDeterminant, Computes) {
    const auto decomposition = num::linalg::cholesky_decompose(spd_matrix()).value();

    EXPECT_DOUBLE_EQ(num::linalg::cholesky_determinant(decomposition), 36.0);
}

TEST(CholeskySolve, Solves) {
    const auto matrix = spd_matrix();
    const num::Vector<double> rhs{-20.0, -43.0, 192.0};
    const auto decomposition = num::linalg::cholesky_decompose(matrix).value();
    const auto solution = num::linalg::cholesky_solve(decomposition, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{1.0, 2.0, 3.0});
}

TEST(CholeskySolve, MatchesLuSolve) {
    const auto matrix = spd_matrix();
    const num::Vector<double> rhs{-20.0, -43.0, 192.0};
    const auto cholesky_decomposition = num::linalg::cholesky_decompose(matrix).value();
    const auto cholesky_solution = num::linalg::cholesky_solve(cholesky_decomposition, rhs).value();
    const auto lu_decomposition = num::linalg::lu_decompose(matrix).value();
    const auto lu_solution = num::linalg::lu_solve(lu_decomposition, rhs).value();

    num::test::expect_vector_near(cholesky_solution, lu_solution);
}

TEST(CholeskySolve, ReportsNonFiniteRhsInput) {
    const auto decomposition = num::linalg::cholesky_decompose(spd_matrix()).value();
    const num::Vector<double> non_finite_rhs{1.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    const auto solution = num::linalg::cholesky_solve(decomposition, non_finite_rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(CholeskySolve, ThrowsOnDimensionMismatch) {
    const auto decomposition = num::linalg::cholesky_decompose(spd_matrix()).value();
    const num::Vector<double> rhs{1.0};

    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_solve(decomposition, rhs)),
                 std::invalid_argument);
}

TEST(CholeskyInverse, Computes) {
    const auto matrix = spd_matrix();
    const auto inverse =
        num::linalg::cholesky_inverse(num::linalg::cholesky_decompose(matrix).value());
    const auto matrix_times_inverse = matrix * inverse;

    num::test::expect_matrix_near(matrix_times_inverse,
                                  num::Matrix<double>::identity(matrix.rows()));
}
