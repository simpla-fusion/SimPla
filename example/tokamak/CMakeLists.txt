FILE(GLOB tokamak_SRC ./*.cpp)
ADD_EXECUTABLE(tokamak ${tokamak_SRC})
SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(tokamak SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} ${TBB_LIBRARIES} ${CUDA_LIBRARIES}
        -Wl,--whole-archive toolbox geometry  data  parallel mesh engine model physics
        predefine third_part -Wl,--no-whole-archive   gomp
        )
