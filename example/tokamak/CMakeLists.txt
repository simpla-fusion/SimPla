FILE(GLOB tokamak_SRC ./*.cpp)
ADD_EXECUTABLE(tokamak ${tokamak_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(tokamak SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive utilities    data  parallel mesh engine model physics
        predefine third_part -Wl,--no-whole-archive   ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )
