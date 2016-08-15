add_subdirectory(cpu)

IF (ENABLE_CUDA)
    add_subdirectory(cuda)
ENDIF (ENABLE_CUDA)
