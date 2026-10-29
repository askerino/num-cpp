#include "num/linalg/qr.hpp"

#include <cstddef>
#include <cstdio>
#include <print>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    const num::Matrix<double> a{
        {1.0, 0.0},
        {1.0, 1.0},
        {1.0, 2.0},
    };
    const num::Vector<double> b{1.0, 2.0, 4.0};

    const auto decomposition = num::linalg::qr_decompose(a);
    if (!decomposition) {
        std::println(stderr, "error: {}", num::error_message(decomposition.error()));
        return 1;
    }

    const auto solution = num::linalg::qr_least_squares(*decomposition, b);
    if (!solution) {
        std::println(stderr, "error: {}", num::error_message(solution.error()));
        return 1;
    }
    const auto& coefficients = *solution;

    const auto residual = (a * coefficients - b).norm_l2();

    std::println("A =");
    for (std::size_t row = 0; row < a.rows(); ++row) {
        std::print("[");
        for (std::size_t column = 0; column < a.cols(); ++column) {
            std::print("{:8.3f}", a(row, column));
        }
        std::println(" ]");
    }

    std::print("b = [");
    for (const auto value : b) {
        std::print("{:8.3f}", value);
    }
    std::println(" ]");

    std::println("Fit: y = {:.3f}x + {:.3f}", coefficients[1], coefficients[0]);

    std::println("Residual: ||Ax - b||_2 = {:.3e}", residual);
    return 0;
}
