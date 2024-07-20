# cmake_minimum_required(VERSION 3.26)
# # cmake_policy(SET CMP0057 NEW)
# cmake_policy(SET CMP0074 NEW)

# PROJECT(FRCexample CXX C)

# MESSAGE(STATUS "Using C++ Compiler ${CMAKE_CXX_COMPILER_ID} ")

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MESSAGE("-- ARCHITECTURE: " ${CMAKE_LIBRARY_ARCHITECTURE})

# SET(BUILD_SHARED_LIBS OFF)


ADD_EXECUTABLE(frc frc.cpp main.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/simpla/utils/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/simpla/core/context.cpp
    )
target_include_directories(frc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

