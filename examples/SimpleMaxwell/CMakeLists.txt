FILE(GLOB SimpleMaxwell_SRC ./*.cpp)
ADD_EXECUTABLE(SimpleMaxwell ${SimpleMaxwell_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(SimpleMaxwell SimPlaMain
        ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES}
        -Wl,--whole-archive
        utilities  algebra  parallel mesh engine physics predefine
        geometry geometry_backend
        data     data_backend
        -Wl,--no-whole-archive
        ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )
