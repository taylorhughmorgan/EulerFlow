cmake_minimum_required(VERSION 3.10)
project(MyCProject C CXX)

# GSL required
find_package(GSL REQUIRED)

# set release flags
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -fopt-info-vec=vec_report.txt")

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add executable and source files
set(SOURCES
    src/test_tns.cpp
    src/taylorneumannsedov.cpp
    src/TVNS_coefs.c
)

add_executable(main ${SOURCES})

# Add optimization flags for release build
target_compile_options(main PRIVATE
    $<$<CONFIG:Release>:
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -fopt-info-vec=vec_report.xt
    >
)


# Link the math library
target_link_libraries(main m)
target_link_libraries(main GSL::gsl GSL::gslcblas)