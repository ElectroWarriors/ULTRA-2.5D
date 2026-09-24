cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_C_COMPILER /path/to/.local/lib/gcc-13.3.0/bin/gcc)
# set(CMAKE_CXX_COMPILER /path/to/.local/lib/gcc-13.3.0/bin/g++)
# set(CMAKE_C_COMPILER /opt/rh/devtoolset-11/root/bin/gcc)
# set(CMAKE_CXX_COMPILER /opt/rh/devtoolset-11/root/bin/g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gdwarf-4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
# cmake ..  -DCMAKE_C_COMPILER=/path/to/gcc-13.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/path/to/gcc-13.3.0/bin/g++

set(CMAKE_CXX_STANDARD 17)     
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)       

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # 支持 device-side struct 和跨文件调用
set(CMAKE_CUDA_ARCHITECTURES 80)

set(PROJECT_NAME "cu_placer")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_ABI 0)
# deal with GCC dual ABI issue 
if(NOT CMAKE_CXX_ABI)
    set(CMAKE_CXX_ABI 0 CACHE STRING
        "Choose the value for _GLIBCXX_USE_CXX11_ABI, options are: 0|1."
        FORCE)
endif(NOT CMAKE_CXX_ABI)
message(STATUS "CMAKE_CXX_ABI: _GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI})

# output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# cuda
set(CUDA_TOOLKIT_ROOT_DIR "/path/to/cuda-12.4")
set(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
file(GLOB CUDA_LIBRARIES
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so
)
# BLAS
set(BLAS_LIBRARIES /path/to/.local/lib/blas-3.12.0/lib/libblas.so.3)
# LAPACK
set(LAPACK_LIBRARIES /path/to/.local/lib/lapack-3.10.0/lib64/liblapack.so.3)
# OPENBLAS
set(OPENBLAS_LIBRARIES /path/to/.local/lib/openblas/lib64/libopenblaso.so.0)
# Boost
set(Boost_ROOT_DIR /path/to/.local/lib/boost_python1.82.0)
set(Boost_INCLUDE_DIR ${Boost_ROOT_DIR}/include/)
file(GLOB Boost_LIBRARIES
    ${Boost_ROOT_DIR}/lib/libboost_system.a
    ${Boost_ROOT_DIR}/lib/libboost_graph.a
    ${Boost_ROOT_DIR}/lib/libboost_iostreams.a
)
message("-- BOOST_LIBRARIES: ${BOOST_LIBRARIES}")

# cplex
set(CPLEX_ROOT_DIR /path/to/.local/lib/cplex)
file(GLOB STATIC_LIB 
    ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic/libcplex.a
    ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic/libilocplex.a
    ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic/libconcert.a
    )
find_library(CPLEX_LIBRARIES
    NAMES libcplex12100.so
    PATHS ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_linux
    )
find_package(Threads REQUIRED)


# Jsoncpp
set(JSON_ROOT_DIR /path/to/.local/lib/jsoncpp)
find_library(JSON_LIBRARIES
    NAMES libjsoncpp.so
    PATHS ${JSON_ROOT_DIR}/lib64
    )
message("-- JSON_LIBRARIES: ${JSON_LIBRARIES}")

# spdlog
set(SPDLOG_ROOT_DIR /path/to/.local/lib/spdlog-1.x/)
find_library(SPDLOG_LIBRARIES
    NAMES libspdlog.a
    PATHS ${SPDLOG_ROOT_DIR}/lib64
    )
message("-- SPDLOG_LIBRARIES: ${SPDLOG_LIBRARIES}")

# arma
set(ARMA_ROOT_DIR /path/to/.local/lib/armadillo)
find_library(ARMA_LIBRARIES
    NAMES libarmadillo.so
    PATHS ${ARMA_ROOT_DIR}/lib64
    )
message("-- ARMA_LIBRARIES: ${ARMA_LIBRARIES}")

# BaseMatrixOps
set(BASEMATRIXOPS_ROOT_DIR /path/to/.local/lib)

# optimlib
set(OPTIMLIB_ROOT_DIR /path/to/.local/lib/optimlib)
find_library(OPTIMLIB_LIBRARIES
    NAMES liboptim.so
    PATHS ${OPTIMLIB_ROOT_DIR}
    )
message("-- OPTIMLIB_LIBRARIES: ${OPTIMLIB_LIBRARIES}")

# source file
file(GLOB SOURCES *.h  *.cpp *.cu *.cuh)
file(GLOB EXE_SOURCES main.cpp)


# include third party
set(TO_INCLUDE_PATH 
    ${CUDA_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${CPLEX_ROOT_DIR}/concert/include/
    ${CPLEX_ROOT_DIR}/cplex/include/
    ${JSON_ROOT_DIR}/include/
    ${SPDLOG_ROOT_DIR}/include/
    ${ARMA_ROOT_DIR}/include/
    ${BASEMATRIXOPS_ROOT_DIR}
    ${OPTIMLIB_ROOT_DIR}/include/optim
)
include_directories(${TO_INCLUDE_PATH})


# exe
add_executable(${PROJECT_NAME} ${SOURCES} ${EXE_SOURCES})

# link
set(TO_LINK_LIBS 
    ${Boost_LIBRARIES} 
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${OPENBLAS_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CPLEX_LIBRARIES} 
    ${JSON_LIBRARIES} 
    ${ARMA_LIBRARIES} 
    ${OPTIMLIB_LIBRARIES} 
    ${SPDLOG_LIBRARIES} 
    ${STATIC_LIB}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TO_LINK_LIBS} gomp pthread m dl)