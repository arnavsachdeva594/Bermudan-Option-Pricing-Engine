cmake_minimum_required(VERSION 3.20)
project(bermudan_pricer LANGUAGES CXX)

# ---- Language standard --------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Third-party dependencies (fetched, not vendored) -------------------
# Eigen 3.4 for the least-squares regression in Longstaff-Schwartz.
# Catch2 v3 for the test suite.
include(FetchContent)

# Eigen is header-only; we only need its headers, so skip its own tests/blas.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)
FetchContent_MakeAvailable(Eigen3)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---- The pricing library ------------------------------------------------
# Header-only for now (payoff/params). Concrete .cpp pricers get added to
# PRICER_SOURCES as later stages land, turning this into a compiled library.
set(PRICER_SOURCES
    src/blackscholes.cpp     # stage 2
    # src/binomial.cpp       # stage 3
    # src/lsm.cpp            # stage 4
    # src/greeks.cpp         # stage 6
)

if(PRICER_SOURCES)
    add_library(pricer STATIC ${PRICER_SOURCES})
else()
    add_library(pricer INTERFACE)
endif()

if(PRICER_SOURCES)
    target_include_directories(pricer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(pricer PUBLIC Eigen3::Eigen)
    find_package(Threads REQUIRED)
    target_link_libraries(pricer PUBLIC Threads::Threads)
else()
    target_include_directories(pricer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(pricer INTERFACE Eigen3::Eigen)
endif()

# ---- Tests --------------------------------------------------------------
enable_testing()
include(CTest)
include(Catch)

add_executable(unit_tests
    tests/test_payoff.cpp
    tests/test_blackscholes.cpp
    tests/test_gbm.cpp
)
target_link_libraries(unit_tests PRIVATE pricer Catch2::Catch2WithMain)
catch_discover_tests(unit_tests)
