include(FetchContent)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif ()

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.18.0
        FIND_PACKAGE_ARGS NAMES GTest
)
FetchContent_MakeAvailable(googletest)

add_executable(num_tests
        core/test_matrix.cpp
        core/test_vector.cpp
        linalg/test_error.cpp
        linalg/test_lu.cpp
)

target_link_libraries(num_tests PRIVATE
        num
        num_project_options
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(num_tests)
