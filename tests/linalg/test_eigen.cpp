#include "num/linalg/eigen.hpp"
#include "support/comparison.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {

num::Matrix<double> symmetric_matrix() {
    return {
        {4.0, 1.0, -2.0, 2.0}, {1.0, 2.0, 0.0, 1.0}, {-2.0, 0.0, 3.0, -1.0}, {2.0, 1.0, -1.0, 5.0}};
}

void expect_reconstruction(const num::Matrix<double>& matrix,
                           const num::linalg::EigenDecomposition<double>& decomposition) {
    const auto lambda = num::Matrix<double>::from_diagonal(decomposition.values);
    num::test::expect_matrix_near(
        decomposition.vectors * lambda * decomposition.vectors.transpose(), matrix);
}

} // namespace

TEST(EigenDecompose, ComputesKnownDecomposition) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();
    const double inv_sqrt2 = 1.0 / std::numbers::sqrt2;

    num::test::expect_vector_near(decomposition.values, num::Vector<double>{1.0, 3.0});
    num::test::expect_matrix_near(
        decomposition.vectors,
        num::Matrix<double>{{inv_sqrt2, inv_sqrt2}, {-inv_sqrt2, inv_sqrt2}});
}

TEST(EigenDecompose, SatisfiesAEqualsVLambdaVt) {
    const auto matrix = symmetric_matrix();
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    expect_reconstruction(matrix, decomposition);
}

TEST(EigenDecompose, ReturnsEigenvaluesAscending) {
    const num::Matrix<double> matrix{{1.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 2.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    num::test::expect_vector_near(decomposition.values, num::Vector<double>{1.0, 2.0, 3.0});
}

TEST(EigenDecompose, ReturnsOrthonormalEigenvectors) {
    const auto decomposition = num::linalg::eigen_decompose(symmetric_matrix()).value();

    num::test::expect_orthonormal_columns(decomposition.vectors);
}

TEST(EigenDecompose, OrientsEigenvectors) {
    const auto decomposition = num::linalg::eigen_decompose(symmetric_matrix()).value();

    num::test::expect_oriented_columns(decomposition.vectors);
}

TEST(EigenDecompose, HandlesSingleElementMatrix) {
    const num::Matrix<double> matrix{{4.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    num::test::expect_vector_near(decomposition.values, num::Vector<double>{4.0});
    num::test::expect_matrix_near(decomposition.vectors, num::Matrix<double>{{1.0}});
}

TEST(EigenDecompose, HandlesTinyScales) {
    const num::Matrix<double> matrix{{1e-200, 1e-200}, {1e-200, 3e-200}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    const auto scaled_values = decomposition.values / 1e-200;
    num::test::expect_vector_near(
        scaled_values, num::Vector<double>{2.0 - std::numbers::sqrt2, 2.0 + std::numbers::sqrt2});
}

TEST(EigenDecompose, HandlesExtremeScaleDifferences) {
    const num::Matrix<double> matrix{{1e300, 1e-300}, {1e-300, -1e300}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    ASSERT_EQ(decomposition.values.size(), 2);
    EXPECT_TRUE(std::isfinite(decomposition.values[0]));
    EXPECT_TRUE(std::isfinite(decomposition.values[1]));
    EXPECT_DOUBLE_EQ(decomposition.values[0], -1e300);
    EXPECT_DOUBLE_EQ(decomposition.values[1], 1e300);
}

TEST(EigenDecompose, HandlesDiagonalMatrix) {
    const num::Matrix<double> matrix{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    num::test::expect_vector_near(decomposition.values, num::Vector<double>{1.0, 2.0, 3.0});
    num::test::expect_matrix_near(decomposition.vectors, num::Matrix<double>::identity(3));
}

TEST(EigenDecompose, HandlesRepeatedEigenvalues) {
    const num::Matrix<double> matrix{{2.0, 0.0}, {0.0, 2.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    num::test::expect_vector_near(decomposition.values, num::Vector<double>{2.0, 2.0});
    expect_reconstruction(matrix, decomposition);
}

TEST(EigenDecompose, ReportsNonFiniteInput) {
    const num::Matrix<double> matrix{{1.0, std::numeric_limits<double>::infinity()},
                                     {std::numeric_limits<double>::infinity(), 1.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_finite_input);
}

TEST(EigenDecompose, ReportsNonSymmetricInput) {
    const num::Matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::non_symmetric);
}

TEST(EigenDecompose, ReportsNonConvergence) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};
    const auto decomposition = num::linalg::eigen_decompose(matrix, 0.0, 0);

    ASSERT_FALSE(decomposition.has_value());
    EXPECT_EQ(decomposition.error(), num::Error::no_convergence);
}

TEST(EigenDecompose, ThrowsOnNonFiniteTolerance) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};

    EXPECT_THROW(static_cast<void>(num::linalg::eigen_decompose(
                     matrix, std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(
                     num::linalg::eigen_decompose(matrix, std::numeric_limits<double>::infinity())),
                 std::invalid_argument);
}

TEST(EigenDecompose, ThrowsOnNegativeTolerance) {
    const num::Matrix<double> matrix{{2.0, 1.0}, {1.0, 2.0}};

    EXPECT_THROW(static_cast<void>(num::linalg::eigen_decompose(matrix, -1.0)),
                 std::invalid_argument);
}

TEST(EigenDecompose, ThrowsOnInvalidInputs) {
    const num::Matrix<double> empty;
    const num::Matrix<double> non_square(2, 3);

    EXPECT_THROW(static_cast<void>(num::linalg::eigen_decompose(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(num::linalg::eigen_decompose(non_square)),
                 std::invalid_argument);
}

TEST(EigenDecompose, SupportsFloat) {
    const num::Matrix<float> matrix{{2.0F, 1.0F}, {1.0F, 2.0F}};
    const auto decomposition = num::linalg::eigen_decompose(matrix).value();

    EXPECT_NEAR(decomposition.values[0], 1.0F, num::test::float_tolerance);
    EXPECT_NEAR(decomposition.values[1], 3.0F, num::test::float_tolerance);
}
