#pragma once

#include <cstdint>
#include <string_view>

namespace num {

/// @addtogroup core
/// @{

enum class Error : std::uint8_t {
    non_finite_input,
    singular_matrix,
    non_positive_definite,
};

[[nodiscard]] constexpr std::string_view error_message(Error error) noexcept {
    switch (error) {
    case Error::non_finite_input:
        return "input contains a non-finite value";
    case Error::singular_matrix:
        return "matrix is singular";
    case Error::non_positive_definite:
        return "matrix is not positive definite";
    }
    return "unknown error";
}

/// @}

} // namespace num
