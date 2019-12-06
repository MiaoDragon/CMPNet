cmake_minimum_required(VERSION 2.8)
project(CMPNet)
# add torch and ompl paths here
#set(CMAKE_PREFIX_PATH /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release ${CMAKE_PREFIX_PATH})
list(APPEND CMAKE_PREFIX_PATH
    ${PROJECT_SOURCE_DIR}/../deps/libtorch
    /usr/local/cuda-9.0)
#set(CMAKE_MODULE_PATH /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release ${CMAKE_MODULE_PATH})
#message("CMAKE_MODULE_PATH" "${CMAKE_MODULE_PATH}")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CUDNN_INCLUDE_DIR "/usr/local/cuda-9.0/include")
#set(CUDNN_LIBRARY "/usr/local/cuda-9.0/lib64/libcudnn.so")
#set(CUDNN_LIBRARY "/usr/local/cuda-9.0/lib64")

set(INCLUDE_DIRECTORIES include)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.0")
add_definitions(-O3 )

find_package(ompl HINTS /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release)

find_package(Eigen3)
message("found ompl library path: " ${OMPL_LIBRARIES} ", " ${OMPLAPP_LIBRARIES})

set(OMPL_LIBRARIES /usr/local/lib/libompl.so)
set(OMPLAPP_LIBRARIES
    /usr/local/lib/libompl_app_base.so
    /usr/local/lib/libompl_app.so)

set(OMPL_INCLUDE_DIRS
    /usr/local/include/ompl
    /usr/local/include/omplapp)
#message("ompl inlucde path: " ${OMPL_INCLUDE_DIRS})
# reset OMPL_LIBRARIES to /usr/local/lib
#set(OMPL_LIBRARIES /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release/lib/libompl.so)
#set(OMPLAPP_LIBRARIES
#    /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release/lib/libompl_app_base.so
#    /home/arclabdl1/ompl/omplapp-1.4.2-Source/build/Release/lib/libompl_app.so)


find_package(Torch REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OMPL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        )
message("found torch library path: " "${TORCH_LIBRARIES}")
set(TORCH_LIBRARIES
    torch
    caffe2_library
    caffe2_gpu_library
    /home/arclabdl1/YLmiao/CMPNet/deps/libtorch/lib/libc10.so
    /usr/local/cuda-9.0/lib64/stubs/libcuda.so
    /usr/local/cuda-9.0/lib64/libnvrtc.so
    /usr/local/cuda-9.0/lib64/libcudart.so
    /home/arclabdl1/YLmiao/CMPNet/deps/libtorch/lib/libc10_cuda.so)



set(INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/include
    ${OMPL_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS})

set(LIB_SOURCE
    src/mpnet_planner.cpp
    src/home_ompl.cpp
)
set(EXEC_SOURCE
    src/home_ompl.cpp)

add_library(${PROJECT_NAME} SHARED ${LIB_SOURCE})

message("OMPLAPPBASE" ${OMPLAPPBASE_LIBRARIES})
#add_library(sst_module SHARED
#    ${PROJECT_SOURCE_DIR}/src/python_wrapper.cpp)
#target_include_directories(${PROJECT_NAME} ${INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OMPLAPP_LIBRARIES} ${OMPL_LIBRARIES} ${TORCH_LIBRARIES})
# Don't prepend wrapper library name with lib and add to Python libs.

add_executable(home_ompl ${EXEC_SOURCE})
#target_include_directories(home_ompl ${PROJECT_NAME})
target_link_libraries(home_ompl ${OMPLAPP_LIBRARIES} ${OMPL_LIBRARIES}  ${TORCH_LIBRARIES} ${PROJECT_NAME})

set_property(TARGET home_ompl PROPERTY CXX_STANDARD 11)
