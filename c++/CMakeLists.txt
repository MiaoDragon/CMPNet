cmake_minimum_required(VERSION 2.8)
project(CMPNet)
# add torch and ompl paths here
list(APPEND CMAKE_PREFIX_PATH
    ${PROJECT_SOURCE_DIR}/../deps/libtorch)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CUDNN_INCLUDE_DIR "/usr/local/cuda-9.0/include")
#set(CUDNN_LIBRARY "/usr/local/cuda-9.0/lib64")
#set(CUDNN_LIBRARY "/usr/local/cuda-9.0/lib64")

set(INCLUDE_DIRECTORIES include)

add_definitions(-O3 )

IF(APPLE)
   set(CMAKE_MACOSX_RPATH 1)
ENDIF(APPLE)


find_package(ompl)
find_package(Torch REQUIRED)
include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${OMPL_INCLUDE_DIRS}
        )


set(SOURCE_FILES
    src/mpnet_planner.cpp
    src/home_ompl.cpp
)

add_library(${PROJECT_NAME} STATIC
    ${SOURCE_FILES})

#add_library(sst_module SHARED
#    ${PROJECT_SOURCE_DIR}/src/python_wrapper.cpp)

target_link_libraries(${PROJECT_NAME} ${OMPL_LIBRARIES} ${OMPLAPP_LIBRARIES} ${TORCH_LIBRARIES})

# Don't prepend wrapper library name with lib and add to Python libs.
#set_target_properties(sst_module PROPERTIES
#        PREFIX "_"
#        SUFFIX ".so"
#        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/sparse_rrt
#        )

add_executable(home_ompl ${SOURCE_FILES})
target_link_libraries(home_ompl ${OMPL_LIBRARIES} ${OMPLAPP_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET home_ompl PROPERTY CXX_STANDARD 11)
