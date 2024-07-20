FILE(GLOB tokamak_SRC ./*.cpp)
ADD_EXECUTABLE(tokamak ${tokamak_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
MESSAGE(STATUS ${OPENMP_LIBRARIES} )
TARGET_LINK_LIBRARIES(tokamak SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive
        utilities  algebra    parallel mesh engine physics geometry geometry_backend   data data_backend
        predefine
        -Wl,--no-whole-archive
        ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )
