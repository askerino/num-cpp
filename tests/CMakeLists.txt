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
