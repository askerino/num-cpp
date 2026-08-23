#include "num/core/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

template <typename T>
concept MatrixElement = requires { typename num::Matrix<T>; };

static_assert(MatrixElement<float>);
static_assert(MatrixElement<double>);
static_assert(MatrixElement<long double>);

static_assert(!MatrixElement<int>);
static_assert(!MatrixElement<bool>);

TEST(MatrixTest, Constructs) {
    const num::Matrix<double> matrix(2, 3);

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.size(), 6);
    for (const auto value : matrix) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }

    const num::Matrix<double> from_elements{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    EXPECT_EQ(from_elements.rows(), 2);
    EXPECT_EQ(from_elements.cols(), 3);
    EXPECT_EQ(from_elements.size(), 6);
    EXPECT_DOUBLE_EQ(from_elements(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(from_elements(1, 2), 6.0);
}

TEST(MatrixTest, RejectsRaggedRows) {
    EXPECT_THROW((num::Matrix<double>{{1.0, 2.0}, {3.0}}), std::invalid_argument);
    EXPECT_THROW((num::Matrix<double>{{1.0}, {2.0, 3.0}}), std::invalid_argument);
}

TEST(MatrixTest, RejectsSizeOverflow) {
    constexpr std::size_t maximum_size = std::numeric_limits<std::size_t>::max();

    EXPECT_THROW((num::Matrix<double>(maximum_size, 2)), std::length_error);
}

TEST(MatrixTest, CreatesZero) {
    const auto zeros = num::Matrix<double>::zeros(2, 3);

    EXPECT_EQ(zeros.rows(), 2);
    EXPECT_EQ(zeros.cols(), 3);
    EXPECT_EQ(zeros.size(), 6);
    for (const auto value : zeros) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(MatrixTest, CreatesIdentity) {
    const auto identity = num::Matrix<double>::identity(3);

    EXPECT_EQ(identity.rows(), 3);
    EXPECT_EQ(identity.cols(), 3);
    EXPECT_EQ(identity.size(), 9);
    for (std::size_t row = 0; row < identity.rows(); ++row) {
        for (std::size_t column = 0; column < identity.cols(); ++column) {
            EXPECT_DOUBLE_EQ(identity(row, column), row == column ? 1.0 : 0.0);
        }
    }
}

TEST(MatrixTest, AccessesElements) {
    num::Matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};

    EXPECT_DOUBLE_EQ(matrix.at(1, 0), 3.0);
    matrix.at(1, 1) = 9.0;
    EXPECT_DOUBLE_EQ(matrix.at(1, 1), 9.0);
    EXPECT_THROW(static_cast<void>(matrix.at(2, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.at(0, 2)), std::out_of_range);

    const num::Matrix<double> const_matrix{{1.0, 2.0}, {3.0, 4.0}};

    EXPECT_DOUBLE_EQ(const_matrix.at(0, 1), 2.0);
    EXPECT_THROW(static_cast<void>(const_matrix.at(2, 2)), std::out_of_range);
}

TEST(MatrixTest, SupportsContainer) {
    num::Matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    const num::Matrix<double>& const_matrix = matrix;

    auto* data = matrix.data();
    data[0] = 5.0;
    EXPECT_DOUBLE_EQ(matrix(0, 0), 5.0);

    const auto* const_data = const_matrix.data();
    EXPECT_DOUBLE_EQ(const_data[0], 5.0);

    EXPECT_EQ(matrix.size(), 4);
    EXPECT_FALSE(matrix.empty());
    EXPECT_TRUE(num::Matrix<double>{}.empty());
    EXPECT_TRUE(num::Matrix<double>(0, 3).empty());
    EXPECT_TRUE(num::Matrix<double>(3, 0).empty());

    std::ranges::transform(matrix, matrix.begin(), [](double value) { return value * 2.0; });
    EXPECT_DOUBLE_EQ(matrix(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(matrix(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(matrix(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 8.0);
}

TEST(MatrixTest, RejectsShapeMismatch) {
    const num::Matrix<double> matrix{{1.0, 2.0}, {3.0, 4.0}};
    const num::Matrix<double> incompatible{{1.0, 2.0, 3.0}};
    const num::Vector<double> vector{1.0, 2.0, 3.0};

    EXPECT_THROW(static_cast<void>(matrix + incompatible), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix - incompatible), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix * incompatible), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix * vector), std::invalid_argument);
}

TEST(MatrixTest, SupportsArithmetic) {
    const num::Matrix<double> lhs{{1.0, 2.0}, {3.0, 4.0}};
    const num::Matrix<double> rhs{{5.0, 6.0}, {7.0, 8.0}};

    const auto sum = lhs + rhs;
    const auto difference = rhs - lhs;
    const auto left_scaled = 2.0 * lhs;
    const auto right_scaled = lhs * 2.0;
    const auto divided = rhs / 2.0;

    EXPECT_DOUBLE_EQ(sum(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(sum(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(sum(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(sum(1, 1), 12.0);
    EXPECT_DOUBLE_EQ(difference(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(difference(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(difference(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(difference(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(left_scaled(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(left_scaled(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(left_scaled(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(left_scaled(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(right_scaled(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(right_scaled(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(right_scaled(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(right_scaled(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(divided(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(divided(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(divided(1, 0), 3.5);
    EXPECT_DOUBLE_EQ(divided(1, 1), 4.0);
}

TEST(MatrixTest, SupportsUnaryMinus) {
    const num::Matrix<double> matrix{{1.0, -2.0}, {3.0, -4.0}};
    const auto negated = -matrix;

    EXPECT_DOUBLE_EQ(negated(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(negated(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(negated(1, 0), -3.0);
    EXPECT_DOUBLE_EQ(negated(1, 1), 4.0);
}

TEST(MatrixTest, MultipliesMatrices) {
    const num::Matrix<double> lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const num::Matrix<double> rhs{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    const auto result = lhs * rhs;

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 154.0);
}

TEST(MatrixTest, MultipliesVector) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const num::Vector<double> vector{1.0, 0.0, -1.0};
    const auto result = matrix * vector;

    EXPECT_EQ(result.size(), 2);
    EXPECT_DOUBLE_EQ(result[0], -2.0);
    EXPECT_DOUBLE_EQ(result[1], -2.0);
}

TEST(MatrixTest, ComputesNorms) {
    const num::Matrix<double> matrix{{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}};

    EXPECT_DOUBLE_EQ(matrix.norm_l1(), 9.0);
    EXPECT_DOUBLE_EQ(matrix.norm_infinity(), 15.0);
    EXPECT_DOUBLE_EQ(matrix.norm_frobenius(), std::sqrt(91.0));
}

TEST(MatrixTest, ComputesEmptyNorms) {
    const num::Matrix<double> empty;
    const num::Matrix<double> no_rows(0, 3);
    const num::Matrix<double> no_columns(3, 0);

    EXPECT_DOUBLE_EQ(empty.norm_l1(), 0.0);
    EXPECT_DOUBLE_EQ(no_rows.norm_infinity(), 0.0);
    EXPECT_DOUBLE_EQ(no_columns.norm_frobenius(), 0.0);
}

TEST(MatrixTest, AvoidsNormOverflow) {
    const num::Matrix<double> matrix{{1.0e308, 1.0e308}};
    const double norm = matrix.norm_frobenius();

    EXPECT_TRUE(std::isfinite(norm));
    EXPECT_NEAR(norm / 1.0e308, std::sqrt(2.0), 1.0e-12);
}

TEST(MatrixTest, ComputesTransposes) {
    const num::Matrix<double> matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const auto transposed = matrix.transpose();

    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 2);
    EXPECT_EQ(transposed.size(), 6);
    EXPECT_DOUBLE_EQ(transposed(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(transposed(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(transposed(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(transposed(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(transposed(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(transposed(2, 1), 6.0);
}
