FILE(GLOB EmbeddedBoundary_SRC ./*.cpp)
ADD_EXECUTABLE(EmbeddedBoundary ${EmbeddedBoundary_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(EmbeddedBoundary SimPlaMain
        ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES}
        -Wl,--whole-archive
        utilities  algebra    parallel mesh engine geometry physics  data data_backend
        predefine
        -Wl,--no-whole-archive
        ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )