
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

# about this project
project(sparse_benchmarking NONE)
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# set env
#add_definitions(-D DEBUG) # compile time macro definition
include_directories(${PROJECT_SOURCE_DIR}/inc)
link_directories(${PROJECT_SOURCE_DIR}/lib)

message("Source Tree Path: ${PROJECT_SOURCE_DIR}")
message("Binary Tree Path: ${PROJECT_BINARY_DIR}")

add_subdirectory(inc)
add_subdirectory(src)
