#include "num/linalg/lu.hpp"
#include "num/linalg/qr.hpp"
#include "support/comparison.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {
num::Matrix<double> square_matrix() {
    return {{2.0, 1.0, -1.0}, {1.0, 3.0, 2.0}, {3.0, -1.0, 1.0}};
}

num::Vector<double> square_rhs() { return {8.0, 6.0, 5.0}; }

num::Matrix<double> tall_matrix() { return {{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}; }
} // namespace

TEST(QrFactorize, ComputesKnownFactors) {
    const num::Matrix<double> matrix{{1.0, 1.0}, {1.0, -1.0}};
    const auto factorization = num::linalg::qr_factorize(matrix).value();
    const double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    const double sqrt2 = std::numbers::sqrt2;

    num::test::expect_matrix_near(
        factorization.q, num::Matrix<double>{{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}});
    num::test::expect_matrix_near(factorization.r, num::Matrix<double>{{sqrt2, 0.0}, {0.0, sqrt2}});
}

TEST(QrFactorize, SatisfiesAEqualsQR) {
    const auto matrix = square_matrix();
    const auto factorization = num::linalg::qr_factorize(matrix).value();

    num::test::expect_matrix_near(factorization.q * factorization.r, matrix);
}

TEST(QrFactorize, ReturnsOrthonormalColumns) {
    const auto matrix = square_matrix();
    const auto factorization = num::linalg::qr_factorize(matrix).value();
    const auto product = factorization.q.transpose() * factorization.q;

    num::test::expect_matrix_near(product, num::Matrix<double>::identity(factorization.q.cols()));
}

TEST(QrFactorize, ReturnsUpperTriangular) {
    const auto factorization = num::linalg::qr_factorize(square_matrix()).value();

    for (std::size_t row = 1; row < factorization.r.rows(); ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            EXPECT_DOUBLE_EQ(factorization.r(row, column), 0.0);
        }
    }
}

TEST(QrFactorize, SupportsTallMatrices) {
    const auto matrix = tall_matrix();
    const auto factorization = num::linalg::qr_factorize(matrix).value();

    EXPECT_EQ(factorization.q.rows(), matrix.rows());
    EXPECT_EQ(factorization.q.cols(), matrix.cols());
    EXPECT_EQ(factorization.r.rows(), matrix.cols());
    EXPECT_EQ(factorization.r.cols(), matrix.cols());

    num::test::expect_matrix_near(factorization.q * factorization.r, matrix);
    num::test::expect_matrix_near(factorization.q.transpose() * factorization.q,
                                  num::Matrix<double>::identity(matrix.cols()));
}

TEST(QrFactorize, HandlesSingleElementMatrix) {
    const num::Matrix<double> matrix{{4.0}};
    const auto factorization = num::linalg::qr_factorize(matrix).value();

    EXPECT_DOUBLE_EQ(factorization.q(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(factorization.r(0, 0), 4.0);
}

TEST(QrFactorize, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto factorization = num::linalg::qr_factorize(non_finite);

    ASSERT_FALSE(factorization.has_value());
    EXPECT_EQ(factorization.error(), num::Error::non_finite_input);
}

TEST(QrFactorize, ReportsRankDeficientMatrix) {
    const num::Matrix<double> rank_deficient{{1.0, 0.0}, {2.0, 0.0}};
    const auto factorization = num::linalg::qr_factorize(rank_deficient);

    ASSERT_FALSE(factorization.has_value());
    EXPECT_EQ(factorization.error(), num::Error::rank_deficient);
}

TEST(QrFactorize, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> wide(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::qr_factorize(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::qr_factorize(wide)), std::invalid_argument);
}

TEST(QrLeastSquares, SolvesSquareSystem) {
    const auto matrix = square_matrix();
    const auto rhs = square_rhs();
    const auto factorization = num::linalg::qr_factorize(matrix).value();
    const auto solution = num::linalg::qr_least_squares(factorization, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{2.6, 1.8, -1.0});
}

TEST(QrLeastSquares, MatchesLuSolve) {
    const auto matrix = square_matrix();
    const auto rhs = square_rhs();
    const auto qr_factorization = num::linalg::qr_factorize(matrix).value();
    const auto qr_solution = num::linalg::qr_least_squares(qr_factorization, rhs).value();
    const auto lu_factorization = num::linalg::lu_factorize(matrix).value();
    const auto lu_solution = num::linalg::lu_solve(lu_factorization, rhs).value();

    num::test::expect_vector_near(qr_solution, lu_solution);
}

TEST(QrLeastSquares, SolvesOverdeterminedSystem) {
    const auto matrix = tall_matrix();
    const num::Vector<double> rhs{1.0, 2.0, 2.0};
    const auto factorization = num::linalg::qr_factorize(matrix).value();
    const auto solution = num::linalg::qr_least_squares(factorization, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{7.0 / 6.0, 0.5});
}

TEST(QrLeastSquares, ResidualIsOrthogonalToColumns) {
    const auto matrix = tall_matrix();
    const num::Vector<double> rhs{1.0, 2.0, 2.0};
    const auto factorization = num::linalg::qr_factorize(matrix).value();
    const auto solution = num::linalg::qr_least_squares(factorization, rhs).value();
    const auto residual = matrix * solution - rhs;
    const auto normal_residual = matrix.transpose() * residual;

    num::test::expect_vector_near(normal_residual, num::Vector<double>{0.0, 0.0});
}

TEST(QrLeastSquares, ReportsNonFiniteRhsInput) {
    const auto factorization = num::linalg::qr_factorize(tall_matrix()).value();
    const num::Vector<double> rhs{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    const auto solution = num::linalg::qr_least_squares(factorization, rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(QrLeastSquares, ThrowsOnDimensionMismatch) {
    const auto factorization = num::linalg::qr_factorize(tall_matrix()).value();
    const num::Vector<double> rhs{1.0, 2.0};

    EXPECT_THROW(static_cast<void>(num::linalg::qr_least_squares(factorization, rhs)),
                 std::invalid_argument);
}
