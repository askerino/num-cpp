add_library(num_core INTERFACE)
add_library(num::core ALIAS num_core)
target_include_directories(num_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(num_core INTERFACE cxx_std_23)
