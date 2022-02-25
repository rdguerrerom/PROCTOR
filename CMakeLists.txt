cmake_minimum_required(VERSION 3.0)
project(PROCTOR)

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

# Generate python module
add_subdirectory(lib/pybind11)
#pybind11_add_module(python_cpp_example ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

# Allowing to use Eigen 3
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries ("${PROJECT_NAME}_test" Eigen3::Eigen)
