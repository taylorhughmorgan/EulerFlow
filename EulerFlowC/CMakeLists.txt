cmake_minimum_required(VERSION 3.10)
project(MyCProject C)

# GSL required
find_package(GSL REQUIRED)

# set release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/lib)

# Add executable and source files
set(SOURCES
    src/test_tns.c
    src/taylorneumannsedov.c
    src/TVNS_coefs.c
)

add_executable(main ${SOURCES})

# Link the math library
target_link_libraries(main m)
target_link_libraries(main GSL::gsl GSL::gslcblas)