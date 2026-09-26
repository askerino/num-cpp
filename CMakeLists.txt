cmake_minimum_required(VERSION 3.25)

project(num-cpp
        VERSION 0.1.0
        DESCRIPTION "A C++ numerical computing library for learning"
        LANGUAGES CXX
)

set(CMAKE_CXX_EXTENSIONS OFF)

option(NUM_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(NUM_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(NUM_ENABLE_SANITIZERS "Enable ASan + UBSan (Clang only)" OFF)
option(NUM_ENABLE_COVERAGE "Enable code coverage (Clang only)" OFF)
option(NUM_BUILD_DOCS "Build API documentation with Doxygen" OFF)

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
        message(FATAL_ERROR "NUM_ENABLE_SANITIZERS requires Clang")
    endif ()
endif ()

if (NUM_ENABLE_COVERAGE)
    if (NOT NUM_BUILD_TESTS)
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE requires NUM_BUILD_TESTS")
    endif ()
    if (NUM_ENABLE_SANITIZERS)
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE cannot be combined with NUM_ENABLE_SANITIZERS")
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
        message(FATAL_ERROR "NUM_ENABLE_COVERAGE requires Clang")
    endif ()
endif ()

if (NUM_BUILD_TESTS)
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

if (NUM_BUILD_DOCS)
    find_package(Doxygen 1.9.5 REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
            doxygen-awesome-css
            GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
            GIT_TAG v2.5.0
    )
    FetchContent_MakeAvailable(doxygen-awesome-css)
    FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR NUM_DOXYGEN_AWESOME_CSS_DIR)

    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
    set(DOXYGEN_STRIP_FROM_INC_PATH "${PROJECT_SOURCE_DIR}/include")
    set(DOXYGEN_EXCLUDE_SYMBOLS "*::detail")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_SORT_MEMBER_DOCS NO)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_DOT_IMAGE_FORMAT svg)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_MATHJAX_VERSION MathJax_3)
    set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)

    set(DOXYGEN_HTML_EXTRA_STYLESHEET
            "${NUM_DOXYGEN_AWESOME_CSS_DIR}/doxygen-awesome.css"
            "${NUM_DOXYGEN_AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only.css"
    )
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_DISABLE_INDEX NO)
    set(DOXYGEN_FULL_SIDEBAR NO)
    set(DOXYGEN_HTML_COLORSTYLE LIGHT)

    doxygen_add_docs(docs
            include "${PROJECT_SOURCE_DIR}/README.md"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMENT "Generating API documentation"
    )
endif ()
