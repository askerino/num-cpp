add_executable(solve_linear_system solve_linear_system.cpp)
target_link_libraries(solve_linear_system PRIVATE num::linalg num_project_options)

add_executable(least_squares least_squares.cpp)
target_link_libraries(least_squares PRIVATE num::linalg num_project_options)

set_target_properties(solve_linear_system least_squares PROPERTIES COMPILE_WARNING_AS_ERROR ON)

if (NUM_BUILD_TESTS)
    add_test(NAME example.solve_linear_system COMMAND solve_linear_system)
    add_test(NAME example.least_squares COMMAND least_squares)
endif ()
