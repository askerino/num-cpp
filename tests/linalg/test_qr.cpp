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

TEST(QrDecompose, ComputesKnownFactors) {
    const num::Matrix<double> matrix{{1.0, 1.0}, {1.0, -1.0}};
    const auto decomposition = num::linalg::qr_decompose(matrix).value();
    const double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    const double sqrt2 = std::numbers::sqrt2;

    num::test::expect_matrix_near(
        decomposition.q, num::Matrix<double>{{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}});
    num::test::expect_matrix_near(decomposition.r, num::Matrix<double>{{sqrt2, 0.0}, {0.0, sqrt2}});
}

TEST(QrDecompose, SatisfiesAEqualsQR) {
    const auto matrix = square_matrix();
    const auto decomposition = num::linalg::qr_decompose(matrix).value();

    num::test::expect_matrix_near(decomposition.q * decomposition.r, matrix);
}

TEST(QrDecompose, ReturnsOrthonormalColumns) {
    const auto matrix = square_matrix();
    const auto decomposition = num::linalg::qr_decompose(matrix).value();
    const auto product = decomposition.q.transpose() * decomposition.q;

    num::test::expect_matrix_near(product, num::Matrix<double>::identity(decomposition.q.cols()));
}

TEST(QrDecompose, ReturnsUpperTriangular) {
    const auto decomposition = num::linalg::qr_decompose(square_matrix()).value();

    for (std::size_t row = 1; row < decomposition.r.rows(); ++row) {
        for (std::size_t column = 0; column < row; ++column) {
            EXPECT_DOUBLE_EQ(decomposition.r(row, column), 0.0);
        }
    }
}

TEST(QrDecompose, SupportsTallMatrices) {
    const auto matrix = tall_matrix();
    const auto decomposition = num::linalg::qr_decompose(matrix).value();

    EXPECT_EQ(decomposition.q.rows(), matrix.rows());
    EXPECT_EQ(decomposition.q.cols(), matrix.cols());
    EXPECT_EQ(decomposition.r.rows(), matrix.cols());
    EXPECT_EQ(decomposition.r.cols(), matrix.cols());

    num::test::expect_matrix_near(decomposition.q * decomposition.r, matrix);
    num::test::expect_matrix_near(decomposition.q.transpose() * decomposition.q,
                                  num::Matrix<double>::identity(matrix.cols()));
}

TEST(QrDecompose, HandlesSingleElementMatrix) {
    const num::Matrix<double> matrix{{4.0}};
    const auto decomposition = num::linalg::qr_decompose(matrix).value();

    EXPECT_DOUBLE_EQ(decomposition.q(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(decomposition.r(0, 0), 4.0);
}

TEST(QrDecompose, HandlesExtremeScales) {
    const num::Matrix<double> tiny_matrix{{1e-200}};
    const auto tiny_decomposition = num::linalg::qr_decompose(tiny_matrix).value();
    EXPECT_DOUBLE_EQ(tiny_decomposition.q(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(tiny_decomposition.r(0, 0), 1e-200);

    const num::Matrix<double> huge_matrix{{1e200}};
    const auto huge_decomposition = num::linalg::qr_decompose(huge_matrix).value();
    EXPECT_DOUBLE_EQ(huge_decomposition.q(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(huge_decomposition.r(0, 0), 1e200);
}

TEST(QrDecompose, ReportsNonFiniteInput) {
    const num::Matrix<double> non_finite{{1.0, std::numeric_limits<double>::infinity()},
                                         {0.0, 1.0}};
    const auto decomposition = num::linalg::qr_decompose(non_finite);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_finite_input);
}

TEST(QrDecompose, ReportsRankDeficientMatrix) {
    const num::Matrix<double> rank_deficient{{1.0, 0.0}, {2.0, 0.0}};
    const auto decomposition = num::linalg::qr_decompose(rank_deficient);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::rank_deficient);
}

TEST(QrDecompose, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> wide(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::qr_decompose(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::qr_decompose(wide)), std::invalid_argument);
}

TEST(QrLeastSquares, SolvesSquareSystem) {
    const auto matrix = square_matrix();
    const auto rhs = square_rhs();
    const auto decomposition = num::linalg::qr_decompose(matrix).value();
    const auto solution = num::linalg::qr_least_squares(decomposition, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{2.6, 1.8, -1.0});
}

TEST(QrLeastSquares, MatchesLuSolve) {
    const auto matrix = square_matrix();
    const auto rhs = square_rhs();
    const auto qr_decomposition = num::linalg::qr_decompose(matrix).value();
    const auto qr_solution = num::linalg::qr_least_squares(qr_decomposition, rhs).value();
    const auto lu_decomposition = num::linalg::lu_decompose(matrix).value();
    const auto lu_solution = num::linalg::lu_solve(lu_decomposition, rhs).value();

    num::test::expect_vector_near(qr_solution, lu_solution);
}

TEST(QrLeastSquares, SolvesOverdeterminedSystem) {
    const auto matrix = tall_matrix();
    const num::Vector<double> rhs{1.0, 2.0, 2.0};
    const auto decomposition = num::linalg::qr_decompose(matrix).value();
    const auto solution = num::linalg::qr_least_squares(decomposition, rhs).value();

    num::test::expect_vector_near(solution, num::Vector<double>{7.0 / 6.0, 0.5});
}

TEST(QrLeastSquares, ResidualIsOrthogonalToColumns) {
    const auto matrix = tall_matrix();
    const num::Vector<double> rhs{1.0, 2.0, 2.0};
    const auto decomposition = num::linalg::qr_decompose(matrix).value();
    const auto solution = num::linalg::qr_least_squares(decomposition, rhs).value();
    const auto residual = matrix * solution - rhs;
    const auto normal_residual = matrix.transpose() * residual;

    num::test::expect_vector_near(normal_residual, num::Vector<double>{0.0, 0.0});
}

TEST(QrLeastSquares, ReportsNonFiniteRhsInput) {
    const auto decomposition = num::linalg::qr_decompose(tall_matrix()).value();
    const num::Vector<double> rhs{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    const auto solution = num::linalg::qr_least_squares(decomposition, rhs);

    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(QrLeastSquares, ThrowsOnDimensionMismatch) {
    const auto decomposition = num::linalg::qr_decompose(tall_matrix()).value();
    const num::Vector<double> rhs{1.0, 2.0};

    EXPECT_THROW(static_cast<void>(num::linalg::qr_least_squares(decomposition, rhs)),
                 std::invalid_argument);
}
