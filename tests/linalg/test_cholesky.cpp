#include "num/linalg/cholesky.hpp"
#include "num/linalg/lu.hpp"
#include "support/comparison.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
// Symmetric positive definite matrix with a known Cholesky factor (A = LLᵀ).
num::Matrix<double> spd_matrix() {
    return {{4.0, 12.0, -16.0}, {12.0, 37.0, -43.0}, {-16.0, -43.0, 98.0}};
}

// Lower-triangular Cholesky factor L of spd_matrix().
num::Matrix<double> expected_lower() {
    return {{2.0, 0.0, 0.0}, {6.0, 1.0, 0.0}, {-8.0, 5.0, 3.0}};
}
} // namespace

TEST(CholeskyFactorize, ComputesLowerFactor) {
    const auto factorization = num::linalg::cholesky_factorize(spd_matrix()).value();

    num::test::expect_matrix_near(factorization.lower, expected_lower());
}

TEST(CholeskyFactorize, SatisfiesAEqualsLLt) {
    const auto matrix = spd_matrix();
    const auto factorization = num::linalg::cholesky_factorize(matrix).value();
    const auto product = factorization.lower * factorization.lower.transpose();

    num::test::expect_matrix_near(product, matrix);
}

TEST(CholeskyFactorize, ReturnsLowerTriangular) {
    const auto factorization = num::linalg::cholesky_factorize(spd_matrix()).value();

    for (std::size_t row = 0; row < factorization.lower.rows(); ++row) {
        for (std::size_t column = row + 1; column < factorization.lower.cols(); ++column) {
            EXPECT_DOUBLE_EQ(factorization.lower(row, column), 0.0);
        }
    }
}

TEST(CholeskyFactorize, UsesOnlyLowerTriangle) {
    const num::Matrix<double> not_symmetric{{4.0, 100.0}, {1.0, 4.0}};
    const auto factorization = num::linalg::cholesky_factorize(not_symmetric).value();
    const auto product = factorization.lower * factorization.lower.transpose();

    num::test::expect_matrix_near(product, num::Matrix<double>{{4.0, 1.0}, {1.0, 4.0}});
}

TEST(CholeskyFactorize, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto factorization = num::linalg::cholesky_factorize(non_finite);

    ASSERT_FALSE(factorization.has_value());
    EXPECT_EQ(factorization.error(), num::Error::non_finite_input);
}

TEST(CholeskyFactorize, ReportsNonPositiveDefinite) {
    const num::Matrix<double> indefinite{{1.0, 2.0}, {2.0, 1.0}};
    const num::Matrix<double> negative_diagonal{{-1.0, 0.0}, {0.0, 1.0}};
    const num::Matrix<double> zero{{0.0}};

    const auto indefinite_result = num::linalg::cholesky_factorize(indefinite);
    const auto negative_diagonal_result = num::linalg::cholesky_factorize(negative_diagonal);
    const auto zero_result = num::linalg::cholesky_factorize(zero);

    ASSERT_FALSE(indefinite_result.has_value());
    EXPECT_EQ(indefinite_result.error(), num::Error::non_positive_definite);
    ASSERT_FALSE(negative_diagonal_result.has_value());
    EXPECT_EQ(negative_diagonal_result.error(), num::Error::non_positive_definite);
    ASSERT_FALSE(zero_result.has_value());
    EXPECT_EQ(zero_result.error(), num::Error::non_positive_definite);
}

TEST(CholeskyFactorize, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_factorize(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_factorize(non_square)),
                 std::invalid_argument);
}

TEST(CholeskyDeterminant, Computes) {
    const auto factorization = num::linalg::cholesky_factorize(spd_matrix()).value();

    EXPECT_DOUBLE_EQ(num::linalg::cholesky_determinant(factorization), 36.0);
}

TEST(CholeskySolve, Solves) {
    const auto matrix = spd_matrix();
    const num::Vector<double> rhs{-20.0, -43.0, 192.0};
    const auto factorization = num::linalg::cholesky_factorize(matrix).value();
    const auto solution = num::linalg::cholesky_solve(factorization, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{1.0, 2.0, 3.0});
}

TEST(CholeskySolve, MatchesLuSolve) {
    const auto matrix = spd_matrix();
    const num::Vector<double> rhs{-20.0, -43.0, 192.0};
    const auto cholesky_factorization = num::linalg::cholesky_factorize(matrix).value();
    const auto cholesky_solution = num::linalg::cholesky_solve(cholesky_factorization, rhs).value();
    const auto lu_factorization = num::linalg::lu_factorize(matrix).value();
    const auto lu_solution = num::linalg::lu_solve(lu_factorization, rhs).value();

    num::test::expect_vector_near(cholesky_solution, lu_solution);
}

TEST(CholeskySolve, ReportsNonFiniteRhsInput) {
    const auto factorization = num::linalg::cholesky_factorize(spd_matrix()).value();
    const num::Vector<double> non_finite_rhs{1.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
    const auto solution = num::linalg::cholesky_solve(factorization, non_finite_rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(CholeskySolve, ThrowsOnDimensionMismatch) {
    const auto factorization = num::linalg::cholesky_factorize(spd_matrix()).value();
    const num::Vector<double> rhs{1.0};

    EXPECT_THROW(static_cast<void>(num::linalg::cholesky_solve(factorization, rhs)),
                 std::invalid_argument);
}

TEST(CholeskyInverse, Computes) {
    const auto matrix = spd_matrix();
    const auto inverse =
        num::linalg::cholesky_inverse(num::linalg::cholesky_factorize(matrix).value());
    const auto matrix_times_inverse = matrix * inverse;

    num::test::expect_matrix_near(matrix_times_inverse,
                                  num::Matrix<double>::identity(matrix.rows()));
}
