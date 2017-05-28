FILE(GLOB MHDTest_SRC ./*.cpp)

add_executable(MHDTest ${MHDTest_SRC})

TARGET_LINK_LIBRARIES(MHDTest
        #     ${OCE_LIBRARIES}   ${MeshKit_LIBRARIES} ${MOAB_LIBRARIES}
        netcdf
        ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}
        SimPlaMAIN toolbox geometry parallel data mesh engine model physics  third_part
        )