cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER /export/home/jhhuang/.local/lib/gcc-13.3.0/bin/gcc)
set(CMAKE_CXX_COMPILER /export/home/jhhuang/.local/lib/gcc-13.3.0/bin/g++)
# set(CMAKE_C_COMPILER /opt/rh/devtoolset-11/root/bin/gcc)
# set(CMAKE_CXX_COMPILER /opt/rh/devtoolset-11/root/bin/g++)
# cmake -DCMAKE_C_COMPILER=/export/home/jhhuang/.local/lib/gcc-13.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/export/home/jhhuang/.local/lib/gcc-13.3.0/bin/g++ ..
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gdwarf-4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")

set(CMAKE_CXX_STANDARD 17)     
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)    

set(PROJECT_NAME "milp_solver")

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

set(CPLEX_ROOT_DIR /export/home/jhhuang/.local/lib/cplex)
file(GLOB CPLEX_STATIC_LIB 
    ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic/libcplex.a
    ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic/libilocplex.a
    ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic/libconcert.a
    )
find_library(CPLEX_LIBRARIES
    NAMES libcplex12100.so
    PATHS ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_linux
    )
find_package(Threads REQUIRED)


# source file
file(GLOB SOURCES *.h  *.cpp)
file(GLOB EXE_SOURCES main.cpp)

# include third party
set(TO_INCLUDE_PATH 
    ${CPLEX_ROOT_DIR}/concert/include/
    ${CPLEX_ROOT_DIR}/cplex/include/
)
include_directories(${TO_INCLUDE_PATH})

add_executable(${PROJECT_NAME} ${SOURCES} ${EXE_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${CPLEX_LIBRARIES} ${CPLEX_STATIC_LIB} pthread m dl)