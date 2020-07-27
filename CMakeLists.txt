cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
project(ReproBugHDF5 VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wall -O3 -g")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # default is release
endif()

include_directories("${PROJECT_SOURCE_DIR}")
find_package(Legion REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

add_executable(debug_hdf5 debug_hdf5.cpp my_mapper.cpp)
target_include_directories(debug_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(debug_hdf5 PRIVATE Legion::Legion ${HDF5_LIBRARIES})
set_target_properties(debug_hdf5 PROPERTIES ENABLE_EXPORTS ON)
