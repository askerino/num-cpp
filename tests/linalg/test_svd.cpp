#include "num/linalg/svd.hpp"
#include "support/comparison.hpp"

#include <cmath>
#include <cstddef>
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

void expect_orthonormal_columns(const num::Matrix<double>& matrix) {
    num::test::expect_matrix_near(matrix.transpose() * matrix,
                                  num::Matrix<double>::identity(matrix.cols()));
}

void expect_oriented_columns(const num::Matrix<double>& matrix) {
    for (std::size_t column = 0; column < matrix.cols(); ++column) {
        std::size_t largest = 0;
        for (std::size_t row = 1; row < matrix.rows(); ++row) {
            if (std::abs(matrix(row, column)) > std::abs(matrix(largest, column))) {
                largest = row;
            }
        }
        EXPECT_GE(matrix(largest, column), 0.0);
    }
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

    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
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
    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
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
    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, OrientsLeftSingularVectors) {
    const auto tall_decomposition = num::linalg::svd_decompose(tall_matrix()).value();
    const auto wide_decomposition = num::linalg::svd_decompose(wide_matrix()).value();

    expect_oriented_columns(tall_decomposition.u);
    expect_oriented_columns(wide_decomposition.u);
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
    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
}

TEST(SvdDecompose, HandlesZeroColumnMatrix) {
    const num::Matrix<double> matrix{{0.0, 2.0}, {0.0, 0.0}};
    const auto decomposition = num::linalg::svd_decompose(matrix).value();

    EXPECT_NEAR(decomposition.s[0], 2.0, num::test::tolerance);
    EXPECT_NEAR(decomposition.s[1], 0.0, num::test::tolerance);
    expect_reconstruction(matrix, decomposition);
    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
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
    expect_orthonormal_columns(decomposition.u);
    expect_orthonormal_columns(decomposition.v);
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

    EXPECT_NEAR(decomposition.s[0], 3.0F, 1e-4F);
    EXPECT_NEAR(decomposition.s[1], 2.0F, 1e-4F);
}
