#include "num/linalg/qr.hpp"
#include "num/linalg/svd.hpp"
#include "support/comparison.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

num::Matrix<double> tall_matrix() { return {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}; }
num::Matrix<double> wide_matrix() { return {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; }

void expect_reconstruction(const num::Matrix<double>& matrix,
                           const num::linalg::SvdDecomposition<double>& decomposition) {
    const auto sigma = num::Matrix<double>::from_diagonal(decomposition.s);
    num::test::expect_matrix_near(decomposition.u * sigma * decomposition.v.transpose(), matrix);
}

} // namespace

TEST(SvdDecompose, ComputesKnownSingularValues) {
    const num::Matrix<double> matrix{{2.0, 0.0}, {0.0, 3.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    num::test::expect_vector_near(decomposition.s, num::Vector<double>{3.0, 2.0});
}

TEST(SvdDecompose, SatisfiesAEqualsUSigmaVt) {
    const num::Matrix<double> matrix{
        {4.0, 1.0, -2.0}, {1.0, 2.0, 0.0}, {-2.0, 0.0, 3.0}, {2.0, 1.0, -1.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    expect_reconstruction(matrix, decomposition);
}

TEST(SvdDecompose, ReturnsOrthonormalColumns) {
    const auto decomposition = num::linalg::svd_decompose(tall_matrix()).value();

    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, ReturnsSingularValuesInDescendingOrder) {
    const auto decomposition = num::linalg::svd_decompose(tall_matrix()).value();

    EXPECT_GE(decomposition.s[0], decomposition.s[1]);
}

TEST(SvdDecompose, SupportsTallMatrices) {
    const auto matrix = tall_matrix();
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_EQ(decomposition.u.rows(), 3);
    EXPECT_EQ(decomposition.u.cols(), 2);
    EXPECT_EQ(decomposition.s.size(), 2);
    EXPECT_EQ(decomposition.v.rows(), 2);
    EXPECT_EQ(decomposition.v.cols(), 2);

    expect_reconstruction(matrix, decomposition);
    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, SupportsWideMatrices) {
    const auto matrix = wide_matrix();
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_EQ(decomposition.u.rows(), 2);
    EXPECT_EQ(decomposition.u.cols(), 2);
    EXPECT_EQ(decomposition.s.size(), 2);
    EXPECT_EQ(decomposition.v.rows(), 3);
    EXPECT_EQ(decomposition.v.cols(), 2);

    expect_reconstruction(matrix, decomposition);
    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, OrientsLeftSingularVectors) {
    const auto tall_decomposition = num::linalg::svd_decompose(tall_matrix()).value();
    const auto wide_decomposition = num::linalg::svd_decompose(wide_matrix()).value();

    num::test::expect_oriented_columns(tall_decomposition.u);
    num::test::expect_oriented_columns(wide_decomposition.u);
}

TEST(SvdDecompose, PairsRightSingularVectorSignWithLeft) {
    const num::Matrix<double> matrix{{3.0, 0.0}, {0.0, -2.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    num::test::expect_matrix_near(decomposition.u, num::Matrix<double>::identity(2));
    num::test::expect_matrix_near(decomposition.v, num::Matrix<double>{{1.0, 0.0}, {0.0, -1.0}});
    expect_reconstruction(matrix, decomposition);
}

TEST(SvdDecompose, PairsRightSingularVectorSignWithLeftForWideMatrix) {
    const num::Matrix<double> matrix{{3.0, 0.0, 0.0}, {0.0, -2.0, 0.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    num::test::expect_matrix_near(decomposition.u, num::Matrix<double>::identity(2));
    num::test::expect_matrix_near(decomposition.v,
                                  num::Matrix<double>{{1.0, 0.0}, {0.0, -1.0}, {0.0, 0.0}});
    expect_reconstruction(matrix, decomposition);
}

TEST(SvdDecompose, HandlesSingleElementMatrix) {
    const num::Matrix<double> matrix{{-4.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_DOUBLE_EQ(decomposition.s[0], 4.0);
    EXPECT_DOUBLE_EQ(decomposition.u(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(decomposition.v(0, 0), -1.0);
}

TEST(SvdDecompose, HandlesRankDeficientMatrix) {
    const num::Matrix<double> matrix{{1.0, 2.0}, {2.0, 4.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(decomposition.s[0], 5.0, num::test::tolerance);
    EXPECT_NEAR(decomposition.s[1], 0.0, num::test::tolerance);
    expect_reconstruction(matrix, decomposition);
    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, HandlesZeroColumnMatrix) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {0.0, 0.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(decomposition.s[0], 2.0, num::test::tolerance);
    EXPECT_NEAR(decomposition.s[1], 0.0, num::test::tolerance);
    expect_reconstruction(matrix, decomposition);
    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, HandlesZeroMatrix) {
    const num::Matrix<double> matrix(2, 3);
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    ASSERT_EQ(decomposition.s.size(), 2);
    EXPECT_DOUBLE_EQ(decomposition.s[0], 0.0);
    EXPECT_DOUBLE_EQ(decomposition.s[1], 0.0);
    EXPECT_EQ(decomposition.u.rows(), 2);
    EXPECT_EQ(decomposition.u.cols(), 2);
    EXPECT_EQ(decomposition.v.rows(), 3);
    EXPECT_EQ(decomposition.v.cols(), 2);

    expect_reconstruction(matrix, decomposition);
    num::test::expect_orthonormal_columns(decomposition.u);
    num::test::expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, HandlesExtremeScales) {
    const num::Matrix<double> tiny{{1e-200, 0.0}, {0.0, 3e-200}};
    const auto tiny_decomposition = num::linalg::svd_decompose(tiny).value();
    num::test::expect_vector_near(tiny_decomposition.s / 1e-200, num::Vector<double>{3.0, 1.0});

    const num::Matrix<double> huge{{1e300, 0.0}, {0.0, 3e300}};
    const auto huge_decomposition = num::linalg::svd_decompose(huge).value();
    num::test::expect_vector_near(huge_decomposition.s / 1e300, num::Vector<double>{3.0, 1.0});
}

TEST(SvdDecompose, SucceedsWhenAlreadyConverged) {
    const num::Matrix<double> matrix{{1.0, 1.0}, {1.0, -1.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix, 0.0, 0);

    EXPECT_TRUE(decomposition.has_value());
}

TEST(SvdDecompose, ReportsNonFiniteInput) {
    const num::Matrix<double> matrix{{1.0, std::numeric_limits<double>::infinity()}, {0.0, 1.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_finite_input);
}

TEST(SvdDecompose, ReportsNonConvergence) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix, 0.0, 0);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::no_convergence);
}

TEST(SvdDecompose, ThrowsOnNegativeTolerance) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};

    EXPECT_THROW(static_cast<void>(num::linalg::svd_decompose(matrix, -1.0)),
                 std::invalid_argument);
}

TEST(SvdDecompose, ThrowsOnNonFiniteTolerance) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};

    EXPECT_THROW(static_cast<void>(
                     num::linalg::svd_decompose(matrix, std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(
                     num::linalg::svd_decompose(matrix, std::numeric_limits<double>::infinity())),
                 std::invalid_argument);
}

TEST(SvdDecompose, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> empty_rows(0, 3);
    const num::Matrix<double> empty_columns(3, 0);

    EXPECT_THROW(static_cast<void>(num::linalg::svd_decompose(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::svd_decompose(empty_rows)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::svd_decompose(empty_columns)),
                 std::invalid_argument);
}

TEST(SvdDecompose, SupportsFloat) {
    const num::Matrix<float> matrix{{2.0F, 0.0F}, {0.0F, 3.0F}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(decomposition.s[0], 3.0F, num::test::float_tolerance);
    EXPECT_NEAR(decomposition.s[1], 2.0F, num::test::float_tolerance);
}

TEST(SvdRank, CountsSingularValues) {
    const num::Matrix<double> full_rank{{2.0, 0.0}, {0.0, 3.0}};
    const auto full_rank_decomposition = num::linalg::svd_decompose(full_rank).value();

    const num::Matrix<double> rank_deficient{{1.0, 2.0}, {2.0, 4.0}};
    const auto rank_deficient_decomposition = num::linalg::svd_decompose(rank_deficient).value();

    EXPECT_EQ(num::linalg::svd_rank(full_rank_decomposition), 2);
    EXPECT_EQ(num::linalg::svd_rank(rank_deficient_decomposition), 1);
}

TEST(SvdRank, HandlesZeroMatrix) {
    const num::Matrix<double> matrix(2, 3);
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_EQ(num::linalg::svd_rank(decomposition), 0);
}

TEST(SvdRank, SupportsFloat) {
    const num::Matrix<float> matrix{{2.0F, 0.0F}, {0.0F, 3.0F}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_EQ(num::linalg::svd_rank(decomposition), 2);
}

TEST(SvdLeastSquares, SolvesOverdeterminedSystem) {
    const num::Matrix<double> matrix{{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}};
    const num::Vector<double> rhs{1.0, 2.0, 4.0};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs).value();
    num::test::expect_vector_near(solution, num::Vector<double>{5.0 / 6.0, 1.5});
}

TEST(SvdLeastSquares, MatchesQrLeastSquares) {
    const auto matrix = tall_matrix();
    const num::Vector<double> rhs{1.0, -2.0, 3.0};

    const auto qr_solution =
        num::linalg::qr_least_squares(num::linalg::qr_decompose(matrix).value(), rhs).value();
    const auto svd_solution =
        num::linalg::svd_least_squares(num::linalg::svd_decompose(matrix).value(), rhs).value();

    num::test::expect_vector_near(svd_solution, qr_solution);
}

TEST(SvdLeastSquares, SolvesUnderdeterminedSystem) {
    const num::Matrix<double> matrix{{1.0, 1.0}};
    const num::Vector<double> rhs{2.0};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs).value();
    num::test::expect_vector_near(solution, num::Vector<double>{1.0, 1.0});
}

TEST(SvdLeastSquares, SolvesRankDeficientSystem) {
    const num::Matrix<double> matrix{{1.0, 1.0}, {1.0, 1.0}};
    const num::Vector<double> rhs{2.0, 2.0};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs).value();
    num::test::expect_vector_near(solution, num::Vector<double>{1.0, 1.0});
}

TEST(SvdLeastSquares, SolvesInconsistentRankDeficientSystem) {
    const num::Matrix<double> matrix{{1.0, 0.0}, {0.0, 0.0}};
    const num::Vector<double> rhs{1.0, 2.0};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs).value();
    num::test::expect_vector_near(solution, num::Vector<double>{1.0, 0.0});
}

TEST(SvdLeastSquares, ReportsNonFiniteRhsInput) {
    const num::Matrix<double> matrix{{1.0, 0.0}, {0.0, 1.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const num::Vector<double> rhs{1.0, std::numeric_limits<double>::quiet_NaN()};

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs);
    ASSERT_FALSE(solution.has_value());
    EXPECT_EQ(solution.error(), num::Error::non_finite_input);
}

TEST(SvdLeastSquares, ThrowsOnDimensionMismatch) {
    const num::Matrix<double> matrix{{1.0, 0.0}, {0.0, 1.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const num::Vector<double> rhs{1.0, 2.0, 3.0};

    EXPECT_THROW(static_cast<void>(num::linalg::svd_least_squares(decomposition, rhs)),
                 std::invalid_argument);
}

TEST(SvdLeastSquares, SupportsFloat) {
    const num::Matrix<float> matrix{{2.0F, 0.0F}, {0.0F, 3.0F}};
    const num::Vector<float> rhs{4.0F, 9.0F};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    const auto solution = num::linalg::svd_least_squares(decomposition, rhs).value();
    EXPECT_NEAR(solution[0], 2.0F, num::test::float_tolerance);
    EXPECT_NEAR(solution[1], 3.0F, num::test::float_tolerance);
}

TEST(SvdPseudoInverse, Computes) {
    const num::Matrix<double> matrix{{4.0, 7.0}, {2.0, 6.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    num::test::expect_matrix_near(num::linalg::svd_pseudo_inverse(decomposition),
                                  num::Matrix<double>{{0.6, -0.7}, {-0.2, 0.4}});
}

TEST(SvdPseudoInverse, SatisfiesMoorePenroseConditions) {
    const auto matrix = tall_matrix();
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const auto inverse = num::linalg::svd_pseudo_inverse(decomposition);

    num::test::expect_matrix_near(matrix * inverse * matrix, matrix);
    num::test::expect_matrix_near(inverse * matrix * inverse, inverse);
    num::test::expect_matrix_near((matrix * inverse).transpose(), matrix * inverse);
    num::test::expect_matrix_near((inverse * matrix).transpose(), inverse * matrix);
}

TEST(SvdPseudoInverse, SupportsWideMatrices) {
    const auto matrix = wide_matrix();
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const auto inverse = num::linalg::svd_pseudo_inverse(decomposition);

    EXPECT_EQ(inverse.rows(), 3);
    EXPECT_EQ(inverse.cols(), 2);
    num::test::expect_matrix_near(matrix * inverse * matrix, matrix);
}

TEST(SvdPseudoInverse, HandlesRankDeficientMatrix) {
    const num::Matrix<double> matrix{{1.0, 2.0}, {2.0, 4.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    num::test::expect_matrix_near(num::linalg::svd_pseudo_inverse(decomposition),
                                  num::Matrix<double>{{0.04, 0.08}, {0.08, 0.16}});
}

TEST(SvdPseudoInverse, HandlesZeroMatrix) {
    const num::Matrix<double> matrix(2, 3);
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const auto inverse = num::linalg::svd_pseudo_inverse(decomposition);

    EXPECT_EQ(inverse.rows(), 3);
    EXPECT_EQ(inverse.cols(), 2);
    num::test::expect_matrix_near(inverse, num::Matrix<double>(3, 2));
}

TEST(SvdPseudoInverse, SupportsFloat) {
    const num::Matrix<float> matrix{{2.0F, 0.0F}, {0.0F, 3.0F}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();
    const auto inverse = num::linalg::svd_pseudo_inverse(decomposition);

    EXPECT_NEAR(inverse(0, 0), 0.5F, num::test::float_tolerance);
    EXPECT_NEAR(inverse(0, 1), 0.0F, num::test::float_tolerance);
    EXPECT_NEAR(inverse(1, 0), 0.0F, num::test::float_tolerance);
    EXPECT_NEAR(inverse(1, 1), 1.0F / 3.0F, num::test::float_tolerance);
}

TEST(SvdConditionNumber, Computes) {
    const num::Matrix<double> matrix{{4.0, 0.0}, {0.0, 2.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(num::linalg::svd_condition_number(decomposition), 2.0, num::test::tolerance);
}

TEST(SvdConditionNumber, HandlesRankDeficientMatrix) {
    const num::Matrix<double> matrix{{1.0, 2.0}, {2.0, 4.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_TRUE(std::isinf(num::linalg::svd_condition_number(decomposition)));
}

TEST(SvdConditionNumber, HandlesNumericallyRankDeficientMatrix) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_EQ(num::linalg::svd_rank(decomposition), 2);
    EXPECT_TRUE(std::isinf(num::linalg::svd_condition_number(decomposition)));
}

TEST(SvdConditionNumber, HandlesZeroMatrix) {
    const num::Matrix<double> matrix(2, 3);
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_TRUE(std::isinf(num::linalg::svd_condition_number(decomposition)));
}

TEST(SvdConditionNumber, SupportsFloat) {
    const num::Matrix<float> matrix{{4.0F, 0.0F}, {0.0F, 2.0F}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(num::linalg::svd_condition_number(decomposition), 2.0F, num::test::float_tolerance);
}
