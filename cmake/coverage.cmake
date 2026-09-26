set(required_variables
        NUM_COVERAGE_CTEST_EXECUTABLE
        NUM_COVERAGE_TEST_EXECUTABLE
        NUM_COVERAGE_LLVM_PROFDATA_EXECUTABLE
        NUM_COVERAGE_LLVM_COV_EXECUTABLE
        NUM_COVERAGE_SOURCE_DIR
        NUM_COVERAGE_BINARY_DIR
)

foreach (variable IN LISTS required_variables)
    if (NOT DEFINED ${variable} OR "${${variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing required coverage variable: ${variable}")
    endif ()
endforeach ()

set(profile_directory "${NUM_COVERAGE_BINARY_DIR}/coverage-data")
set(raw_profile_pattern "${profile_directory}/num_tests-%p.profraw")
set(indexed_profile "${profile_directory}/num_tests.profdata")

file(REMOVE_RECURSE "${profile_directory}")
file(MAKE_DIRECTORY "${profile_directory}")

set(ctest_command
        "${NUM_COVERAGE_CTEST_EXECUTABLE}"
        --test-dir "${NUM_COVERAGE_BINARY_DIR}"
        --output-on-failure
)
if (DEFINED NUM_COVERAGE_CONFIG AND NOT NUM_COVERAGE_CONFIG STREQUAL "")
    list(APPEND ctest_command --build-config "${NUM_COVERAGE_CONFIG}")
endif ()

execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
        "LLVM_PROFILE_FILE=${raw_profile_pattern}"
        ${ctest_command}
        RESULT_VARIABLE test_result
)
if (NOT test_result EQUAL 0)
    message(FATAL_ERROR "Tests failed while collecting coverage data")
endif ()

file(GLOB raw_profiles LIST_DIRECTORIES false "${profile_directory}/*.profraw")
if (NOT raw_profiles)
    message(FATAL_ERROR "No LLVM raw coverage profiles were generated")
endif ()

execute_process(
        COMMAND "${NUM_COVERAGE_LLVM_PROFDATA_EXECUTABLE}"
        merge
        -sparse
        ${raw_profiles}
        -o "${indexed_profile}"
        RESULT_VARIABLE merge_result
)
if (NOT merge_result EQUAL 0)
    message(FATAL_ERROR "llvm-profdata failed to merge coverage profiles")
endif ()

file(GLOB_RECURSE coverage_sources
        LIST_DIRECTORIES false
        "${NUM_COVERAGE_SOURCE_DIR}/include/num/*.hpp"
)
if (NOT coverage_sources)
    message(FATAL_ERROR "No library headers were found for coverage reporting")
endif ()

execute_process(
        COMMAND "${NUM_COVERAGE_LLVM_COV_EXECUTABLE}"
        report
        "${NUM_COVERAGE_TEST_EXECUTABLE}"
        "-instr-profile=${indexed_profile}"
        --sources
        ${coverage_sources}
        RESULT_VARIABLE report_result
)
if (NOT report_result EQUAL 0)
    message(FATAL_ERROR "llvm-cov failed to generate the coverage report")
endif ()
