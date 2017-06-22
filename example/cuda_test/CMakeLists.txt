FILE(GLOB cuda_test_SRC ./*.cu)
SET(CUDA_NVCC_FLAGS "--std=c++11")
cuda_add_executable(cuda_test ${cuda_test_SRC})


FILE(GLOB clang_test_SRC ./*.cpp)
cuda_add_executable(clang_test ${clang_test_SRC})
IF (CUDA_FOUND)
    set_target_properties(clang_test PROPERTIES COMPILE_FLAGS "-std=c++14 -xcuda -v")
    target_link_libraries(clang_test ${CUDA_LIBRARIES})
ENDIF (CUDA_FOUND)