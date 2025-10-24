# --- Project Setup ---
cmake_minimum_required(VERSION 3.10)
project(MixedRadixFFT)

# --- Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER clang++)

# =================================================================
# 1. THE LIBRARY (fft_lib)
# =================================================================
# This library will contain all your core FFT and utility code.
add_library(fft_lib
    src/fft_lib.cpp
    src/fft_utils.cpp
)

# Tell CMake that this library's public headers are in 'include/'.
# Any executable that links to 'fft_lib' will automatically get
# this include path.
target_include_directories(fft_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =================================================================
# 2. THE MAIN TEST HARNESS (fft_runner)
# =================================================================
# This is your original main.cpp, which runs the full FFT tests.
add_executable(fft_runner
    app/main.cpp
)

# Link the runner against the library.
target_link_libraries(fft_runner PRIVATE fft_lib)


# =================================================================
# 3. THE UNIT TEST EXECUTABLE (test_utils)
# =================================================================
# This is a new, separate executable for testing small utility functions.
add_executable(test_utils
    test/test_utils.cpp
)

# Link the unit tests against the library.
target_link_libraries(test_utils PRIVATE fft_lib)