add_library(num_linalg INTERFACE)
add_library(num::linalg ALIAS num_linalg)
target_include_directories(num_linalg INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(num_linalg INTERFACE num::core)
