#include "num/linalg/qr.hpp"

#include <cstddef>
#include <cstdio>
#include <print>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    const num::Matrix<double> matrix{
        {1.0, 0.0},
        {1.0, 1.0},
        {1.0, 2.0},
    };
    const num::Vector<double> rhs{1.0, 2.0, 4.0};

    const auto decomposition = num::linalg::qr_decompose(matrix);
    if (!decomposition) {
        std::println(stderr, "error: {}", num::error_message(decomposition.error()));
        return 1;
    }

    const auto solution = num::linalg::qr_least_squares(*decomposition, rhs);
    if (!solution) {
        std::println(stderr, "error: {}", num::error_message(solution.error()));
        return 1;
    }

    std::println("A =");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        std::print("[");
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            std::print("{:8.3f}", matrix(row, column));
        }
        std::println(" ]");
    }

    std::print("b = [");
    for (const auto value : rhs) {
        std::print("{:8.3f}", value);
    }
    std::println(" ]");

    std::println("Fit: y = {:.3f}x + {:.3f}", (*solution)[1], (*solution)[0]);

    const auto residual = (matrix * (*solution) - rhs).norm_l2();
    std::println("Residual: ||Ax - b||_2 = {:.3e}", residual);
    return 0;
}
