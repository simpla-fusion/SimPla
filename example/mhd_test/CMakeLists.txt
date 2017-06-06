FILE(GLOB MHDTest_SRC ./*.cpp)

add_executable(MHDTest ${MHDTest_SRC}  )

TARGET_LINK_LIBRARIES(MHDTest SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive  toolbox geometry  data   mesh engine model physics  predefine third_part -Wl,--no-whole-archive
        )