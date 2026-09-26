#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace num {

/// @defgroup core Core
/// @{

/// @brief A dynamically sized vector of floating-point values.
///
/// @note Storage is contiguous, and operator[] does not perform bounds checking.
/// @note `Vector<double>(5)` creates five zero-initialized elements, whereas `Vector<double>{5.0}`
/// creates a single element.
template <std::floating_point T>
class Vector {
  public:
    Vector() = default;
    explicit Vector(std::size_t element_count) : data_(element_count, T{}) {}
    Vector(std::initializer_list<T> values) : data_(values) {}

    [[nodiscard]] static Vector zeros(std::size_t element_count) { return Vector(element_count); }

    [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    /// @throws std::out_of_range If @p index is out of bounds.
    [[nodiscard]] T& at(std::size_t index) {
        require_valid_index(index);
        return data_[index];
    }

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

    /// @throws std::invalid_argument If the vectors have different sizes.
    Vector& operator+=(const Vector& rhs) {
        require_same_size(rhs);
        for (std::size_t index = 0; index < size(); ++index) {
            data_[index] += rhs[index];
        }
        return *this;
    }

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

    /// @throws std::invalid_argument If the vectors have different sizes.
    [[nodiscard]] friend Vector operator+(Vector lhs, const Vector& rhs) {
        lhs += rhs;
        return lhs;
    }

    /// @throws std::invalid_argument If the vectors have different sizes.
    [[nodiscard]] friend Vector operator-(Vector lhs, const Vector& rhs) {
        lhs -= rhs;
        return lhs;
    }

    [[nodiscard]] friend Vector operator-(Vector vector) {
        vector *= T{-1};
        return vector;
    }

    [[nodiscard]] friend Vector operator*(Vector vector, T scalar) {
        vector *= scalar;
        return vector;
    }

    [[nodiscard]] friend Vector operator*(T scalar, Vector vector) {
        vector *= scalar;
        return vector;
    }

    [[nodiscard]] friend Vector operator/(Vector vector, T scalar) {
        vector /= scalar;
        return vector;
    }

    /// @throws std::invalid_argument If the vectors have different sizes.
    [[nodiscard]] T dot(const Vector& rhs) const {
        require_same_size(rhs);
        T result{};
        for (std::size_t index = 0; index < size(); ++index) {
            result += data_[index] * rhs[index];
        }
        return result;
    }

    [[nodiscard]] T norm_l1() const {
        T result{};
        for (const auto element : data_) {
            result += std::abs(element);
        }
        return result;
    }

    [[nodiscard]] T norm_l2() const {
        T result{};
        // Use std::hypot to avoid overflow/underflow when squaring large or small values.
        for (const auto element : data_) {
            result = std::hypot(result, element);
        }
        return result;
    }

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

/// @}

} // namespace num
