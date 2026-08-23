#pragma once

#include <cstdint>
#include <string_view>
namespace num {

/// @brief Identifies an error reported by a linear algebra operation.
enum class LinalgError : std::uint8_t {
    singular_matrix,  ///< The input matrix is singular.
    non_finite_input, ///< The input contains a non-finite value.
};

/// @brief Returns a human-readable message for @p error.
/// @return A string view with static storage duration.
[[nodiscard]] constexpr std::string_view linalg_error_message(LinalgError error) noexcept {
    switch (error) {
    case LinalgError::singular_matrix:
        return "matrix is singular";
    case LinalgError::non_finite_input:
        return "input contains a non-finite value";
    }
    return "unknown linear algebra error";
}

} // namespace num
