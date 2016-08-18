add_subdirectory(openmp)

IF (ENABLE_CUDA)
    add_subdirectory(cuda)
ENDIF (ENABLE_CUDA)
