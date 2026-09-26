#pragma once

#include <cstdint>
#include <string_view>

namespace num {

/// @addtogroup core
/// @{

/// @brief Identifies an error reported by a numerical operation.
enum class Error : std::uint8_t {
    singular_matrix,
    non_finite_input,
};

/// @return A view of a string literal, which has static storage duration.
[[nodiscard]] constexpr std::string_view error_message(Error error) noexcept {
    switch (error) {
    case Error::singular_matrix:
        return "matrix is singular";
    case Error::non_finite_input:
        return "input contains a non-finite value";
    }
    return "unknown error";
}

/// @}

} // namespace num
