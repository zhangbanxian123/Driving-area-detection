cmake_minimum_required(VERSION 3.14)
project(get_clustered_output)

set(CMAKE_CXX_STANDARD 14)

SET(Torch_DIR /home/zdd/Downloads/libtorch/share/cmake/Torch)
find_package(Torch)
find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS system filesystem)
find_package(MLPACK)

include_directories(/home/zdd/Downloads/libtorch/include/torch/csrc/api/include/)
include_directories(/home/zdd/Downloads/libtorch/include/)
# include_directories(${MLPACK_INCLUDE_DIRS})
include_directories(/usr/local/include/)
include_directories(/usr/local/include/mlpack/core/util)
include_directories(/usr/local/include/boost/)


#####################
# add_subdirectory(hello)
# include_directories(./hello)
##########################

# link_directories(/usr/local/libtorch/include/torch/csrc/api/include/)

add_executable(get_clustered_output get_clustered_output.cpp)

message("=======================${MLPACK_INCLUDE_DIRS}")
message("=======================${TORCH_LIBRARIES}")
message("=======================${MLPACK_LIBRARIES}")
message(("======================${HELLO_SRC}"))
target_link_libraries(get_clustered_output ${Boost_LIBRARIES} ${MLPACK_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# cmake_minimum_required(VERSION 3.14)
# project(get_clustered_outpt)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED True)


# # SET(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
# # find_package(Torch REQUIRED)

# find_package(MLPACK)
# find_package(OpenCV REQUIRED)
# # message(OpenCV)
# # enable_testing ()
# # add_library(catch STATIC catch.hpp)
# # add_library(test_catch_tools STATIC test_catch_tools.hpp)
# # include_directories(/usr/local/libtorch/include/torch/csrc/api/include/)
# include_directories(/usr/local/include/)
# include_directories(/usr/local/include/mlpack/core/util)
# # include_directories(${CMAKE_CURRENT_LIST_DIR})
# # include_directories(${PROJECT_SOURCE_DIR})


# # add_definitions(-DCATCH_IMPL -DCATCH_CONFIG_COUNTER -DCATCH_CONFIG_WCHAR 
# #                 -DCATCH_CONFIG_EXTERNAL_INTERFACES
# #                 -DCATCH_TRAP 
# #                 -DCATCH_CONFIG_ENABLE_CHRONO_STRINGMAKER)#-DCATCH_CONFIG_PREFIX_ALL

# add_executable(get_clustered_outpt get_clustered_outpt.cpp)

# target_link_libraries(get_clustered_outpt ${MLPACK_LIBRARIES} ${OpenCV_LIBS})

# # add_test (NAME MyTest COMMAND mlpackDemo)