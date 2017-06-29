FILE(GLOB tokamak_SRC ./*.cpp)

add_executable(tokamak ${tokamak_SRC})

set_target_properties(tokamak PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})


TARGET_LINK_LIBRARIES(tokamak SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive     toolbox geometry  data  parallel mesh engine model physics
        predefine third_part -Wl,--no-whole-archive   gomp
        )
