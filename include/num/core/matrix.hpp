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

/// @addtogroup core
/// @{

template <std::floating_point T>
class Matrix {
  public:
    Matrix() = default;

    /// @throws std::length_error If the requested matrix is too large.
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(checked_element_count(rows, cols), T{}) {}

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

    /// @throws std::length_error If the requested matrix is too large.
    [[nodiscard]] static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols);
    }

    /// @throws std::length_error If the requested matrix is too large.
    [[nodiscard]] static Matrix identity(std::size_t dimension) {
        Matrix result(dimension, dimension);
        for (std::size_t i = 0; i < dimension; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    /// @note Performs no bounds checking; use at() for a checked access.
    [[nodiscard]] T& operator()(std::size_t row, std::size_t column) noexcept {
        return data_[(row * cols_) + column];
    }

    /// @note Performs no bounds checking; use at() for a checked access.
    [[nodiscard]] const T& operator()(std::size_t row, std::size_t column) const noexcept {
        return data_[(row * cols_) + column];
    }

    /// @throws std::out_of_range If either index is out of bounds.
    [[nodiscard]] T& at(std::size_t row, std::size_t column) {
        require_valid_indices(row, column);
        return data_[(row * cols_) + column];
    }

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

    /// @throws std::invalid_argument If the matrices have different shapes.
    Matrix& operator+=(const Matrix& rhs) {
        require_same_shape(rhs);
        for (std::size_t index = 0; index < data_.size(); ++index) {
            data_[index] += rhs.data_[index];
        }
        return *this;
    }

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

    /// @throws std::invalid_argument If the matrices have different shapes.
    [[nodiscard]] friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
        lhs += rhs;
        return lhs;
    }

    /// @throws std::invalid_argument If the matrices have different shapes.
    [[nodiscard]] friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
        lhs -= rhs;
        return lhs;
    }

    [[nodiscard]] friend Matrix operator-(Matrix matrix) {
        matrix *= T{-1};
        return matrix;
    }

    [[nodiscard]] friend Matrix operator*(Matrix matrix, T scalar) {
        matrix *= scalar;
        return matrix;
    }

    [[nodiscard]] friend Matrix operator*(T scalar, Matrix matrix) {
        matrix *= scalar;
        return matrix;
    }

    [[nodiscard]] friend Matrix operator/(Matrix matrix, T scalar) {
        matrix /= scalar;
        return matrix;
    }

    /// @throws std::invalid_argument If the inner dimensions do not match.
    /// @throws std::length_error If the resulting matrix is too large.
    [[nodiscard]] friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
        if (lhs.cols_ != rhs.rows_) {
            throw std::invalid_argument(std::format(
                "matrix multiplication size mismatch: lhs has {} columns, rhs has {} rows",
                lhs.cols_, rhs.rows_));
        }
        Matrix result(lhs.rows_, rhs.cols_);
        for (std::size_t i = 0; i < lhs.rows_; ++i) {
            for (std::size_t j = 0; j < rhs.cols_; ++j) {
                T sum{};
                for (std::size_t k = 0; k < lhs.cols_; ++k) {
                    sum += lhs(i, k) * rhs(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /// @throws std::invalid_argument If the matrix column count and vector size differ.
    [[nodiscard]] friend Vector<T> operator*(const Matrix& lhs, const Vector<T>& rhs) {
        if (lhs.cols_ != rhs.size()) {
            throw std::invalid_argument(
                std::format("matrix-vector multiplication size mismatch: matrix "
                            "has {} columns, vector has {} elements",
                            lhs.cols_, rhs.size()));
        }
        Vector<T> result(lhs.rows_);
        for (std::size_t i = 0; i < lhs.rows_; ++i) {
            T sum{};
            for (std::size_t j = 0; j < lhs.cols_; ++j) {
                sum += lhs(i, j) * rhs[j];
            }
            result[i] = sum;
        }
        return result;
    }

    [[nodiscard]] T norm_l1() const {
        T result{};
        for (std::size_t j = 0; j < cols_; ++j) {
            T column_sum{};
            for (std::size_t i = 0; i < rows_; ++i) {
                column_sum += std::abs((*this)(i, j));
            }
            if (column_sum > result) {
                result = column_sum;
            }
        }
        return result;
    }

    [[nodiscard]] T norm_infinity() const {
        T result{};
        for (std::size_t i = 0; i < rows_; ++i) {
            T row_sum{};
            for (std::size_t j = 0; j < cols_; ++j) {
                row_sum += std::abs((*this)(i, j));
            }
            if (row_sum > result) {
                result = row_sum;
            }
        }
        return result;
    }

    /// @note Squaring the elements may overflow for very large values or underflow for very small
    /// values.
    [[nodiscard]] T norm_frobenius() const {
        T sum_of_squares{};
        for (const auto element : data_) {
            sum_of_squares += element * element;
        }
        return std::sqrt(sum_of_squares);
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t i = 0; i < cols_; ++i) {
            for (std::size_t j = 0; j < rows_; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    /// @note The empty (0x0) matrix is symmetric, as the symmetry condition is vacuously satisfied.
    /// @note The comparison is exact, so a matrix computed with floating-point operations may not
    /// be symmetric even when it is symmetric in exact arithmetic.
    [[nodiscard]] bool is_symmetric() const {
        if (rows_ != cols_) {
            return false;
        }
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = i + 1; j < cols_; ++j) {
                if ((*this)(i, j) != (*this)(j, i)) {
                    return false;
                }
            }
        }
        return true;
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

/// @}

} // namespace num
