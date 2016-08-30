add_subdirectory(openmp)

IF (ENABLE_CUDA)
    add_subdirectory(cuda)
ENDIF (ENABLE_CUDA)

IF (ENABLE_OPENACC)
    add_subdirectory(openacc)
ENDIF (ENABLE_OPENACC)