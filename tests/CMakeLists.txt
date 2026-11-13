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

include(GoogleTest)

function(num_add_test target library)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE
            num::${library}
            num_project_options
            GTest::gtest
            GTest::gtest_main
    )
    set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    gtest_discover_tests(${target} TEST_PREFIX "${library}." PROPERTIES LABELS ${library})
    set_property(GLOBAL APPEND PROPERTY NUM_TEST_TARGETS ${target})
endfunction()

num_add_test(num_core_tests core
        core/test_error.cpp
        core/test_matrix.cpp
        core/test_vector.cpp
)

num_add_test(num_linalg_tests linalg
        linalg/test_lu.cpp
        linalg/test_cholesky.cpp
        linalg/test_qr.cpp
        linalg/test_eigen.cpp
        linalg/test_svd.cpp
)

if (NUM_ENABLE_COVERAGE)
    get_property(NUM_TEST_TARGETS GLOBAL PROPERTY NUM_TEST_TARGETS)
    set(NUM_TEST_EXECUTABLES "")
    foreach (target IN LISTS NUM_TEST_TARGETS)
        list(APPEND NUM_TEST_EXECUTABLES "$<TARGET_FILE:${target}>")
    endforeach ()

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
            "-DNUM_COVERAGE_TEST_EXECUTABLES=${NUM_TEST_EXECUTABLES}"
            "-DNUM_COVERAGE_LLVM_PROFDATA_EXECUTABLE=${NUM_LLVM_PROFDATA_EXECUTABLE}"
            "-DNUM_COVERAGE_LLVM_COV_EXECUTABLE=${NUM_LLVM_COV_EXECUTABLE}"
            "-DNUM_COVERAGE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DNUM_COVERAGE_BINARY_DIR=${PROJECT_BINARY_DIR}"
            "-DNUM_COVERAGE_CONFIG=$<CONFIG>"
            -P "${PROJECT_SOURCE_DIR}/cmake/coverage.cmake"
            DEPENDS ${NUM_TEST_TARGETS}
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            USES_TERMINAL
            VERBATIM
            COMMENT "Running tests and generating coverage report"
    )
endif ()
