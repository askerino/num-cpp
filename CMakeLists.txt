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

function(num_add_library library)
    set(include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_library(num_${library} INTERFACE)
    add_library(num::${library} ALIAS num_${library})
    target_include_directories(num_${library} INTERFACE "${include_dir}")
    set_property(GLOBAL APPEND PROPERTY NUM_INCLUDE_DIRS "${include_dir}")
endfunction()

add_subdirectory(libs/core)
add_subdirectory(libs/linalg)

add_library(num_project_options INTERFACE)
target_compile_options(num_project_options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /W4 /permissive->
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
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

if (NUM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (NUM_BUILD_DOCS)
    add_subdirectory(docs)
endif ()
