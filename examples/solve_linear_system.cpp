#include "num/linalg/lu.hpp"

#include <cstddef>
#include <print>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
    const num::Matrix<double> matrix{
        {2.0, 1.0, -1.0},
        {1.0, 3.0, 2.0},
        {3.0, -1.0, 1.0},
    };
    const num::Vector<double> rhs{8.0, 6.0, 5.0};

    const auto factorization = num::lu_factorize(matrix);
    if (!factorization) {
        std::println(stderr, "error: {}", num::linalg_error_message(factorization.error()));
        return 1;
    }
    const auto solution = num::lu_solve(*factorization, rhs);
    if (!solution) {
        std::println(stderr, "error: {}", num::linalg_error_message(solution.error()));
        return 1;
    }
    const auto& solution_values = *solution;

    std::println("Solving Ax = b");
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

    std::print("x = [");
    for (const auto value : solution_values) {
        std::print("{:8.3f}", value);
    }
    std::println(" ]");

    const auto residual = ((matrix * solution_values) - rhs).norm_l2();
    std::println("Residual: ||Ax - b||_2 = {:.3e}", residual);
    return 0;
}
