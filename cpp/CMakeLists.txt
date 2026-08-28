cmake_minimum_required(VERSION 3.16)
project(localvol CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

# Absolute path to the shared data directory (../data relative to cpp/):
# golden values, bundled surfaces.
set(LOCALVOL_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")

# ---- library ---------------------------------------------------------------
add_library(localvol
    src/market.cpp
    src/tridiag.cpp
    src/surface.cpp
    src/dupire.cpp
    src/black_scholes.cpp
    src/pde.cpp
    src/mc.cpp
)
target_include_directories(localvol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- demo ------------------------------------------------------------------
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE localvol)
target_compile_definitions(demo PRIVATE LOCALVOL_DATA_DIR="${LOCALVOL_DATA_DIR}")

# ---- tests -----------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)

add_executable(localvol_tests
    tests/test_tridiag.cpp
    tests/test_black_scholes.cpp
    tests/test_surface.cpp
    tests/test_dupire.cpp
    tests/test_pde.cpp
    tests/test_mc.cpp
    tests/test_golden.cpp
)
target_link_libraries(localvol_tests PRIVATE localvol GTest::gtest GTest::gtest_main)
target_compile_definitions(localvol_tests PRIVATE LOCALVOL_DATA_DIR="${LOCALVOL_DATA_DIR}")

include(GoogleTest)
gtest_discover_tests(localvol_tests)
