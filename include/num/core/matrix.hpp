#pragma once

#include "num/core/vector.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief A dynamically sized matrix of floating-point values.
///
/// @note Storage is contiguous and row-major, and operator() does not perform bounds checking.
template <std::floating_point T>
class Matrix {
  public:
    /// @brief Creates an empty matrix with dimensions zero by zero.
    Matrix() = default;

    /// @brief Creates a zero-initialized matrix with dimensions @p rows by @p cols.
    /// @throws std::length_error If the requested matrix is too large.
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), T{}) {}

    /// @brief Creates a matrix from the initializer list @p values.
    /// @throws std::invalid_argument If the rows have different lengths.
    /// @throws std::length_error If the requested matrix is too large.
    Matrix(std::initializer_list<std::initializer_list<T>> values)
        : rows_(values.size()), cols_(values.size() != 0 ? values.begin()->size() : 0),
          data_(checked_element_count(rows_, cols_), T{}) {
        std::size_t row = 0;
        for (const auto& row_values : values) {
            if (row_values.size() != cols_) {
                throw std::invalid_argument(
                    std::format("matrix row size mismatch: row {} has {}, expected {}", row,
                                row_values.size(), cols_));
            }
            std::size_t column = 0;
            for (const auto value : row_values) {
                (*this)(row, column) = value;
                ++column;
            }
            ++row;
        }
    }

    /// @brief Creates a zero-initialized matrix with dimensions @p rows by @p cols.
    /// @throws std::length_error If the requested matrix is too large.
    [[nodiscard]] static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols);
    }

    /// @brief Creates an identity matrix with dimensions @p dimension by @p dimension.
    /// @throws std::length_error If the requested matrix is too large.
    [[nodiscard]] static Matrix identity(std::size_t dimension) {
        Matrix result(dimension, dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            result(index, index) = T{1};
        }
        return result;
    }

    [[nodiscard]] T& operator()(std::size_t row, std::size_t column) noexcept {
        return data_[(row * cols_) + column];
    }
    [[nodiscard]] const T& operator()(std::size_t row, std::size_t column) const noexcept {
        return data_[(row * cols_) + column];
    }

    /// @brief Returns the element at @p row and @p column with bounds checking.
    /// @throws std::out_of_range If either index is out of bounds.
    [[nodiscard]] T& at(std::size_t row, std::size_t column) {
        require_valid_indices(row, column);
        return data_[(row * cols_) + column];
    }

    /// @brief Returns the element at @p row and @p column with bounds checking.
    /// @throws std::out_of_range If either index is out of bounds.
    [[nodiscard]] const T& at(std::size_t row, std::size_t column) const {
        require_valid_indices(row, column);
        return data_[(row * cols_) + column];
    }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    /// @brief Adds @p rhs element-wise.
    /// @throws std::invalid_argument If the matrices have different shapes.
    Matrix& operator+=(const Matrix& rhs) {
        require_same_shape(rhs);
        for (std::size_t index = 0; index < data_.size(); ++index) {
            data_[index] += rhs.data_[index];
        }
        return *this;
    }

    /// @brief Subtracts @p rhs element-wise.
    /// @throws std::invalid_argument If the matrices have different shapes.
    Matrix& operator-=(const Matrix& rhs) {
        require_same_shape(rhs);
        for (std::size_t index = 0; index < data_.size(); ++index) {
            data_[index] -= rhs.data_[index];
        }
        return *this;
    }

    Matrix& operator*=(T scalar) {
        for (auto& element : data_) {
            element *= scalar;
        }
        return *this;
    }

    Matrix& operator/=(T scalar) {
        for (auto& element : data_) {
            element /= scalar;
        }
        return *this;
    }

    /// @brief Computes the matrix product with @p rhs.
    /// @throws std::invalid_argument If the inner dimensions do not match.
    [[nodiscard]] Matrix operator*(const Matrix& rhs) const {
        if (cols_ != rhs.rows_) {
            throw std::invalid_argument(std::format(
                "matrix multiplication size mismatch: lhs has {} columns, rhs has {} rows", cols_,
                rhs.rows_));
        }
        Matrix result(rows_, rhs.cols_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t column = 0; column < rhs.cols_; ++column) {
                T sum{};
                for (std::size_t inner = 0; inner < cols_; ++inner) {
                    sum += (*this)(row, inner) * rhs(inner, column);
                }
                result(row, column) = sum;
            }
        }
        return result;
    }

    /// @brief Computes the matrix-vector product with @p rhs.
    /// @throws std::invalid_argument If the matrix column count and vector size differ.
    [[nodiscard]] Vector<T> operator*(const Vector<T>& rhs) const {
        if (cols_ != rhs.size()) {
            throw std::invalid_argument(
                std::format("matrix-vector multiplication size mismatch: matrix "
                            "has {} columns, vector has {} elements",
                            cols_, rhs.size()));
        }
        Vector<T> result(rows_);
        for (std::size_t row = 0; row < rows_; ++row) {
            T sum{};
            for (std::size_t column = 0; column < cols_; ++column) {
                sum += (*this)(row, column) * rhs[column];
            }
            result[row] = sum;
        }
        return result;
    }

    /// @brief Returns the induced L1 norm, the largest absolute column sum.
    [[nodiscard]] T norm_l1() const {
        T result{};
        for (std::size_t column = 0; column < cols_; ++column) {
            T column_sum{};
            for (std::size_t row = 0; row < rows_; ++row) {
                column_sum += std::abs((*this)(row, column));
            }
            if (column_sum > result) {
                result = column_sum;
            }
        }
        return result;
    }

    /// @brief Returns the induced infinity norm, the largest absolute row sum.
    [[nodiscard]] T norm_infinity() const {
        T result{};
        for (std::size_t row = 0; row < rows_; ++row) {
            T row_sum{};
            for (std::size_t column = 0; column < cols_; ++column) {
                row_sum += std::abs((*this)(row, column));
            }
            if (row_sum > result) {
                result = row_sum;
            }
        }
        return result;
    }

    /// @brief Returns the Frobenius norm, the square root of the sum of the squares of the
    /// elements.
    [[nodiscard]] T norm_frobenius() const {
        T result{};
        for (const auto element : data_) {
            result = std::hypot(result, element);
        }
        return result;
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t column = 0; column < cols_; ++column) {
                // NOLINTNEXTLINE(readability-suspicious-call-argument)
                result(column, row) = (*this)(row, column);
            }
        }
        return result;
    }

  private:
    [[nodiscard]] static std::size_t checked_element_count(std::size_t rows, std::size_t cols) {
        if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::length_error(std::format("matrix dimensions too large: {}x{}", rows, cols));
        }
        return rows * cols;
    }

    void require_valid_indices(std::size_t row, std::size_t column) const {
        if (row >= rows_ || column >= cols_) {
            throw std::out_of_range(
                std::format("matrix index out of range: index ({}, {}), shape {}x{}", row, column,
                            rows_, cols_));
        }
    }

    void require_same_shape(const Matrix& rhs) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            throw std::invalid_argument(std::format("matrix shape mismatch: lhs {}x{}, rhs {}x{}",
                                                    rows_, cols_, rhs.rows_, rhs.cols_));
        }
    }

    std::size_t rows_{};
    std::size_t cols_{};
    std::vector<T> data_;
};

/// @brief Adds two matrices element-wise.
/// @throws std::invalid_argument If the matrices have different shapes.
template <std::floating_point T>
[[nodiscard]] Matrix<T> operator+(Matrix<T> lhs, const Matrix<T>& rhs) {
    lhs += rhs;
    return lhs;
}

/// @brief Subtracts two matrices element-wise.
/// @throws std::invalid_argument If the matrices have different shapes.
template <std::floating_point T>
[[nodiscard]] Matrix<T> operator-(Matrix<T> lhs, const Matrix<T>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator-(Matrix<T> matrix) {
    matrix *= T{-1};
    return matrix;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator*(Matrix<T> matrix, T scalar) {
    matrix *= scalar;
    return matrix;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator*(T scalar, Matrix<T> matrix) {
    matrix *= scalar;
    return matrix;
}

template <std::floating_point T>
[[nodiscard]] Matrix<T> operator/(Matrix<T> matrix, T scalar) {
    matrix /= scalar;
    return matrix;
}

} // namespace num
