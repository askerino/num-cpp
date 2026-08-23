#include "num/core/vector.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

template <typename T>
concept VectorElement = requires { typename num::Vector<T>; };

static_assert(VectorElement<float>);
static_assert(VectorElement<double>);
static_assert(VectorElement<long double>);

static_assert(!VectorElement<int>);
static_assert(!VectorElement<bool>);

TEST(VectorTest, Constructs) {
    const num::Vector<double> vector(5);

    EXPECT_EQ(vector.size(), 5);
    for (const auto value : vector) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }

    const num::Vector<double> from_elements{5.0};

    EXPECT_EQ(from_elements.size(), 1);
    EXPECT_DOUBLE_EQ(from_elements[0], 5.0);
}

TEST(VectorTest, CreatesZero) {
    const auto zeros = num::Vector<double>::zeros(3);

    EXPECT_EQ(zeros.size(), 3);
    for (const auto value : zeros) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(VectorTest, AccessesElements) {
    num::Vector<double> vector{1.0, 2.0, 3.0};

    EXPECT_DOUBLE_EQ(vector.at(1), 2.0);
    vector.at(2) = 9.0;
    EXPECT_DOUBLE_EQ(vector.at(2), 9.0);
    EXPECT_THROW(static_cast<void>(vector.at(3)), std::out_of_range);

    const num::Vector<double> const_vector{1.0, 2.0, 3.0};

    EXPECT_DOUBLE_EQ(const_vector.at(0), 1.0);
    EXPECT_THROW(static_cast<void>(const_vector.at(3)), std::out_of_range);
}

TEST(VectorTest, SupportsContainer) {
    num::Vector<double> vector{1.0, 2.0, 3.0};
    const num::Vector<double>& const_vector = vector;

    auto* data = vector.data();
    data[1] = 4.0;
    EXPECT_DOUBLE_EQ(vector[1], 4.0);

    const auto* const_data = const_vector.data();
    EXPECT_DOUBLE_EQ(const_data[1], 4.0);

    EXPECT_EQ(vector.size(), 3);
    EXPECT_FALSE(vector.empty());
    EXPECT_TRUE(num::Vector<double>{}.empty());

    std::ranges::transform(vector, vector.begin(), [](double value) { return value * 2.0; });
    EXPECT_DOUBLE_EQ(vector[0], 2.0);
    EXPECT_DOUBLE_EQ(vector[1], 8.0);
    EXPECT_DOUBLE_EQ(vector[2], 6.0);
}

TEST(VectorTest, RejectsSizeMismatch) {
    const num::Vector<double> short_vector{1.0, 2.0};
    const num::Vector<double> long_vector{1.0, 2.0, 3.0};

    EXPECT_THROW(static_cast<void>(short_vector + long_vector), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(short_vector - long_vector), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(short_vector.dot(long_vector)), std::invalid_argument);
}

TEST(VectorTest, SupportsArithmetic) {
    const num::Vector<double> lhs{1.0, 2.0, 3.0};
    const num::Vector<double> rhs{4.0, 5.0, 6.0};

    const auto sum = lhs + rhs;
    const auto difference = rhs - lhs;
    const auto left_scaled = 2.0 * lhs;
    const auto right_scaled = lhs * 2.0;
    const auto divided = rhs / 2.0;

    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[1], 7.0);
    EXPECT_DOUBLE_EQ(sum[2], 9.0);
    EXPECT_DOUBLE_EQ(difference[0], 3.0);
    EXPECT_DOUBLE_EQ(difference[1], 3.0);
    EXPECT_DOUBLE_EQ(difference[2], 3.0);
    EXPECT_DOUBLE_EQ(left_scaled[0], 2.0);
    EXPECT_DOUBLE_EQ(left_scaled[1], 4.0);
    EXPECT_DOUBLE_EQ(left_scaled[2], 6.0);
    EXPECT_DOUBLE_EQ(right_scaled[0], 2.0);
    EXPECT_DOUBLE_EQ(right_scaled[1], 4.0);
    EXPECT_DOUBLE_EQ(right_scaled[2], 6.0);
    EXPECT_DOUBLE_EQ(divided[0], 2.0);
    EXPECT_DOUBLE_EQ(divided[1], 2.5);
    EXPECT_DOUBLE_EQ(divided[2], 3.0);
    EXPECT_DOUBLE_EQ(lhs.dot(rhs), 32.0);
}

TEST(VectorTest, SupportsUnaryMinus) {
    const num::Vector<double> vector{1.0, -2.0, 3.0};
    const auto negated = -vector;

    EXPECT_DOUBLE_EQ(negated[0], -1.0);
    EXPECT_DOUBLE_EQ(negated[1], 2.0);
    EXPECT_DOUBLE_EQ(negated[2], -3.0);
}

TEST(VectorTest, ComputesNorms) {
    const num::Vector<double> vector{-3.0, 4.0};

    EXPECT_DOUBLE_EQ(vector.norm_l1(), 7.0);
    EXPECT_DOUBLE_EQ(vector.norm_l2(), 5.0);
    EXPECT_DOUBLE_EQ(vector.norm_infinity(), 4.0);
}

TEST(VectorTest, ComputesEmptyNorms) {
    const num::Vector<double> empty;

    EXPECT_DOUBLE_EQ(empty.norm_l1(), 0.0);
    EXPECT_DOUBLE_EQ(empty.norm_l2(), 0.0);
    EXPECT_DOUBLE_EQ(empty.norm_infinity(), 0.0);
}

TEST(VectorTest, AvoidsNormOverflow) {
    const num::Vector<double> vector{1.0e308, 1.0e308};
    const double norm = vector.norm_l2();

    EXPECT_TRUE(std::isfinite(norm));
    EXPECT_NEAR(norm / 1.0e308, std::sqrt(2.0), 1.0e-12);
}
