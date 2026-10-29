#include "num/linalg/lu.hpp"

#include <cstddef>
#include <cstdio>
#include <print>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    const num::Matrix<double> a{
        {2.0, 1.0, -1.0},
        {1.0, 3.0, 2.0},
        {3.0, -1.0, 1.0},
    };
    const num::Vector<double> b{8.0, 6.0, 5.0};

    const auto decomposition = num::linalg::lu_decompose(a);
    if (!decomposition) {
        std::println(stderr, "error: {}", num::error_message(decomposition.error()));
        return 1;
    }
    const auto solution = num::linalg::lu_solve(*decomposition, b);
    if (!solution) {
        std::println(stderr, "error: {}", num::error_message(solution.error()));
        return 1;
    }
    const auto& x = *solution;

    const auto residual = ((a * x) - b).norm_l2();

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

    std::print("Solution: x = [");
    for (const auto value : x) {
        std::print("{:8.3f}", value);
    }
    std::println(" ]");

    std::println("Residual: ||Ax - b||_2 = {:.3e}", residual);
    return 0;
}
