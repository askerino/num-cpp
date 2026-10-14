#pragma once

#include <cstdint>
#include <string_view>

namespace num {

/// @namespace num
/// @brief Root namespace of the library.
///
/// Formulas are written in plain text with 0-based indices to match the code, e.g. `a_ij` for the
/// element in row `i` and column `j`, and `sum_i` for a sum over all `i`.

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
