#include "num/core/error.hpp"
#include "num/core/matrix.hpp"
#include "num/core/vector.hpp"
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
    for (std::size_t i = 0; i < a.rows(); ++i) {
        std::println("{::8.3f}", a.row(i));
    }

    std::println("b = {::8.3f}", b);

    std::println("Solution: x = {::8.3f}", x);

    std::println("Residual: ||Ax - b||_2 = {:.3e}", residual);
    return 0;
}
