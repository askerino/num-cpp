add_executable(solve_linear_system solve_linear_system.cpp)
target_link_libraries(solve_linear_system PRIVATE num num_project_options)

add_executable(least_squares least_squares.cpp)
target_link_libraries(least_squares PRIVATE num num_project_options)

if (NUM_BUILD_TESTS)
    add_test(NAME Examples.SolveLinearSystem COMMAND solve_linear_system)
    add_test(NAME Examples.LeastSquares COMMAND least_squares)
endif ()
