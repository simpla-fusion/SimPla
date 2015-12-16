# create a plugin that adds a reader to the ParaView GUI
# it is added in the file dialog when doing opens/saves.


#IF (${BUILD_VTK_READER})

IF (${PARAVIEW_ROOT})

    SET(CMAKE_PREFIX_PATH ${PARAVIEW_ROOT})

    FIND_PACKAGE(ParaView REQUIRED)
    INCLUDE(${PARAVIEW_USE_FILE})
    ADD_PARAVIEW_PLUGIN(AMRSimPlaReader "1.0"
            SERVER_MANAGER_XML simpla_amr_reader.xml
            SERVER_MANAGER_SOURCES vtkAMRSimPlaReader.cpp
            )

    ADD_PARAVIEW_PLUGIN(AMRSimPlaParticlesReader "1.0"
            SERVER_MANAGER_XML simpla_amr_particles_reader.xml
            SERVER_MANAGER_SOURCES vtkAMRSimPlaParticlesReader.cpp
            )


    TARGET_LINK_LIBRARIES(AMRSimPlaReader ${HDF5_C_LIBRARIES})
    TARGET_LINK_LIBRARIES(AMRSimPlaParticlesReader ${HDF5_C_LIBRARIES})

ENDIF (${PARAVIEW_ROOT})