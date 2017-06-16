FILE(GLOB tokamak_SRC ./*.cpp)

add_executable(tokamak ${tokamak_SRC})


TARGET_LINK_LIBRARIES(tokamak SimPlaMain
        netcdf ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive     toolbox geometry  data  parallel mesh engine model physics  predefine third_part -Wl,--no-whole-archive  ${SAMRAI_LIBRARIES}
        )
