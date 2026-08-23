cmake_minimum_required(VERSION 3.25)

project(num-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

option(NUM_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(NUM_ENABLE_COVERAGE "Enable coverage" OFF)
option(BUILD_TESTING "Build tests" OFF)
option(NUM_BUILD_EXAMPLES "Build examples" OFF)

add_library(num INTERFACE)
add_library(num::num ALIAS num)
target_include_directories(num INTERFACE "include")
target_compile_features(num INTERFACE cxx_std_23)

add_library(num_project_options INTERFACE)
target_compile_options(num_project_options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /W4 /permissive- /WX>
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Werror>
)

if (NUM_ENABLE_SANITIZERS)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(num_project_options INTERFACE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
        )
        target_link_options(num_project_options INTERFACE
                -fsanitize=address,undefined
        )
    else ()
        message(FATAL_ERROR "NUM_ENABLE_SANITIZERS requires Clang on Linux")
    endif ()
endif ()

if (NUM_ENABLE_COVERAGE)
    if (NUM_ENABLE_SANITIZERS)
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE cannot be combined with NUM_ENABLE_SANITIZERS")
    endif ()
    if (NOT BUILD_TESTING)
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE requires BUILD_TESTING")
    endif ()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(num_project_options INTERFACE
                -fprofile-instr-generate
                -fcoverage-mapping
                -O0
                -g
        )
        target_link_options(num_project_options INTERFACE
                -fprofile-instr-generate
        )
    else ()
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE requires Clang on Linux")
    endif ()
endif ()

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (NUM_ENABLE_COVERAGE)
    file(REAL_PATH "${CMAKE_CXX_COMPILER}" NUM_CLANG_COMPILER_REAL_PATH)
    get_filename_component(NUM_CLANG_BIN_DIR "${NUM_CLANG_COMPILER_REAL_PATH}" DIRECTORY)
    find_program(NUM_LLVM_PROFDATA_EXECUTABLE
            NAMES llvm-profdata
            HINTS "${NUM_CLANG_BIN_DIR}"
            NAMES_PER_DIR
            REQUIRED
    )
    find_program(NUM_LLVM_COV_EXECUTABLE
            NAMES llvm-cov
            HINTS "${NUM_CLANG_BIN_DIR}"
            NAMES_PER_DIR
            REQUIRED
    )

    add_custom_target(coverage
            COMMAND "${CMAKE_COMMAND}"
            "-DNUM_COVERAGE_CTEST_EXECUTABLE=${CMAKE_CTEST_COMMAND}"
            "-DNUM_COVERAGE_TEST_EXECUTABLE=$<TARGET_FILE:num_tests>"
            "-DNUM_COVERAGE_LLVM_PROFDATA_EXECUTABLE=${NUM_LLVM_PROFDATA_EXECUTABLE}"
            "-DNUM_COVERAGE_LLVM_COV_EXECUTABLE=${NUM_LLVM_COV_EXECUTABLE}"
            "-DNUM_COVERAGE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DNUM_COVERAGE_BINARY_DIR=${PROJECT_BINARY_DIR}"
            "-DNUM_COVERAGE_CONFIG=$<CONFIG>"
            -P "${PROJECT_SOURCE_DIR}/cmake/coverage.cmake"
            DEPENDS num_tests
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            USES_TERMINAL
            VERBATIM
            COMMENT "Running tests and generating coverage report"
    )
endif ()

if (NUM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
