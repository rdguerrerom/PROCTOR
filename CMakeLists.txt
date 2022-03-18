cmake_minimum_required(VERSION 3.0)
project(PROCTOR)
# To be activated later.
# foollowing the advice at:
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#set (FFTW_DIR "/usr/local/Cellar/fftw/3.3.10")
#find_path (FFTW_DIR fftw3.h)
#INCLUDE_DIRECTORIES(${FFTW_DIR})
#link_directories("${FFTW_DIR}")
#TARGET_LINK_LIBRARIES(libfftw3-3 libfftw3f-3 libfftw3l-3)

#https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
#target_compile_features(PROCTOR PUBLIC cxx_std_17)
set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
SET(SOURCE_DIR "src/PROCTOR")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
#SET(SOURCES "${SOURCE_DIR}/level0.cpp")

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cpp"
    "${TEST_DIR}/test_level0.cpp")

# Generate a test executable
include_directories(lib/Catch2/include)
add_executable("${PROJECT_NAME}_test" ${TESTS})
#https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
target_compile_features("${PROJECT_NAME}_test" PUBLIC cxx_std_17)

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(PROCTOR ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

# Allowing to use Eigen 3
target_link_libraries ("${PROJECT_NAME}_test" Eigen3::Eigen)
