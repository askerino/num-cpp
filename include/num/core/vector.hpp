#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief A dynamically sized vector of floating-point values.
///
/// @note Storage is contiguous, and operator[] does not perform bounds checking.
/// @note `Vector<double>(5)` creates five zero-initialized elements, whereas `Vector<double>{5.0}`
/// creates a single element.
template <std::floating_point T>
class Vector {
  public:
    /// @brief Creates an empty vector.
    Vector() = default;

    /// @brief Creates a vector of @p element_count zero-initialized elements.
    explicit Vector(std::size_t element_count) : data_(element_count, T{}) {}

    /// @brief Creates a vector from the initializer list @p values.
    Vector(std::initializer_list<T> values) : data_(values) {}

    [[nodiscard]] static Vector zeros(std::size_t element_count) { return Vector(element_count); }

    [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    /// @brief Returns the element at @p index with bounds checking.
    /// @throws std::out_of_range If @p index is out of bounds.
    [[nodiscard]] T& at(std::size_t index) {
        require_valid_index(index);
        return data_[index];
    }

    /// @brief Returns the element at @p index with bounds checking.
    /// @throws std::out_of_range If @p index is out of bounds.
    [[nodiscard]] const T& at(std::size_t index) const {
        require_valid_index(index);
        return data_[index];
    }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    /// @brief Adds @p rhs element-wise.
    /// @throws std::invalid_argument If the vectors have different sizes.
    Vector& operator+=(const Vector& rhs) {
        require_same_size(rhs);
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] += rhs[index];
        }
        return *this;
    }

    /// @brief Subtracts @p rhs element-wise.
    /// @throws std::invalid_argument If the vectors have different sizes.
    Vector& operator-=(const Vector& rhs) {
        require_same_size(rhs);
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] -= rhs[index];
        }
        return *this;
    }

    Vector& operator*=(T scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
        return *this;
    }

    Vector& operator/=(T scalar) {
        for (auto& element : data_) {
            element /= scalar;
        }
        return *this;
    }

    /// @brief Computes the dot product with @p rhs.
    /// @throws std::invalid_argument If the vectors have different sizes.
    [[nodiscard]] T dot(const Vector& rhs) const {
        require_same_size(rhs);
        T result{};
        for (std::size_t index = 0; index < size(); ++index) {
            result += data_[index] * rhs[index];
        }
        return result;
    }

    /// @brief Returns the L1 norm, the sum of the absolute values of the elements.
    [[nodiscard]] T norm_l1() const {
        T result{};
        for (const auto element : data_) {
            result += std::abs(element);
        }
        return result;
    }

    /// @brief Returns the L2 (Euclidean) norm, the square root of the sum of the squares of the
    /// elements.
    [[nodiscard]] T norm_l2() const {
        T result{};
        for (const auto element : data_) {
            result = std::hypot(result, element);
        }
        return result;
    }

    /// @brief Returns the infinity norm, the largest absolute value among the elements.
    [[nodiscard]] T norm_infinity() const {
        T result{};
        for (const auto element : data_) {
            const T magnitude = std::abs(element);
            if (magnitude > result) {
                result = magnitude;
            }
        }
        return result;
    }

  private:
    void require_valid_index(std::size_t index) const {
        if (index >= size()) {
            throw std::out_of_range(
                std::format("vector index out of range: index {}, size {}", index, size()));
        }
    }

    void require_same_size(const Vector& rhs) const {
        if (size() != rhs.size()) {
            throw std::invalid_argument(
                std::format("vector size mismatch: lhs {}, rhs {}", size(), rhs.size()));
        }
    }

    std::vector<T> data_;
};

/// @brief Adds two vectors element-wise.
/// @throws std::invalid_argument If the vectors have different sizes.
template <std::floating_point T>
[[nodiscard]] Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs) {
    lhs += rhs;
    return lhs;
}

/// @brief Subtracts two vectors element-wise.
/// @throws std::invalid_argument If the vectors have different sizes.
template <std::floating_point T>
[[nodiscard]] Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator-(Vector<T> vector) {
    vector *= T{-1};
    return vector;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator*(Vector<T> vector, T scalar) {
    vector *= scalar;
    return vector;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator*(T scalar, Vector<T> vector) {
    vector *= scalar;
    return vector;
}

template <std::floating_point T>
[[nodiscard]] Vector<T> operator/(Vector<T> vector, T scalar) {
    vector /= scalar;
    return vector;
}

} // namespace num
