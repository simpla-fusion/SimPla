FILE(GLOB MultiDomain_SRC ./*.cpp)
ADD_EXECUTABLE(MultiDomain ${MultiDomain_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(MultiDomain SimPlaMain
          ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES}
        -Wl,--whole-archive
        utilities  algebra    parallel mesh engine physics geometry geometry_backend   data data_backend
        predefine
        -Wl,--no-whole-archive
        ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )
