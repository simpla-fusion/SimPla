cmake_minimum_required(VERSION 2.8)
SET(CMAKE_VERBOSE_MAKEFILE ON)

FIND_PACKAGE(CUDA 7.5 REQUIRED)
IF (CUDA_FOUND)
    ADD_DEFINITIONS(-DUSE_CUDA -DENABLE_SOA -DNUMBER_OF_THREADS_PER_BLOCK=256)
    SET(CUDA_NVCC_FLAGS -DENABLE_SOA --maxrregcount=32)
    FILE(GLOB_RECURSE sp_boris_yee_SRC ./*.cu )
        CUDA_ADD_EXECUTABLE(sp_boris_yee  ${sp_boris_yee_SRC} )
#       TARGET_LINK_LIBRARIES(sp_boris_yee    cuda  sp_api_cuda io data_model ${CUDA_LIBRARIES})        

ENDIF (CUDA_FOUND)
