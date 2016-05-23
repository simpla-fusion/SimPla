SET(CHOMBO_DIR "/work/build/Chombo-3.2")
SET(CHOMBO_CONFIG "2d.Linux.64.mpicxx.gfortran.DEBUG.OPT.MPI")
SET(CHOMBO_CONFIG_3d "3d.Linux.64.mpicxx.gfortran.DEBUG.OPT.MPI")

FIND_PACKAGE(Chombo)

MESSAGE("CHOMOBO INCLUDE:" "${CHOMBO_INCLUDE_DIRS}")
MESSAGE("CHOMOBO LIBRARIES:" "${CHOMBO_LIBRARIES}")
FILE(GLOB_RECURSE test_chombo_SRC *.cpp)
add_executable(test_chombo
        ebamrGodunov.cpp
        EBImplicitIntegratorFactory.cpp
        EBPlanarShockIBC.cpp
        EBPatchPolytropic.cpp
        EBAMRGLoadBalance.cpp
        EBTGAStrategy.cpp
        EBExplosionIBCFactory.cpp
        ModianoIBCFactory.cpp
        EBImplicitIntegrationStrategy.cpp
        EBPatchPolytropicFactory.cpp
        EBCrankNicholsonStrategy.cpp
        SlipWallViscousTensorDomainBC.cpp
        EBAMRGodunovGeomFactory.cpp
        GodunovGeom.cpp
        EBPlanarShockIBCFactory.cpp
        EBExplosionIBC.cpp
        ModianoIBC.cpp
        SmoothBumpIBC.cpp
        SquareCylinderIF.cpp
        EBCNSVortexIBC.cpp
        EBConsBackwardEulerIntegrator.cpp
        WedgeIF.cpp
        EBExplosionF.ChF
        SmoothBumpF.ChF
        EBSolidF.ChF
        EBCNSVortexF.ChF
        ModianoF.ChF
        EBPatchPolytropicF.ChF
        EBPlanarShockF.ChF
        )
target_compile_definitions(test_chombo PUBLIC ${CHOMBO_DEFINES})
target_include_directories(test_chombo PUBLIC ${CHOMBO_INCLUDE_DIRS})
target_link_libraries(test_chombo ${CHOMBO_LIBRARIES} ${MPI_LIBRARIES} ${HDF5_LIBRARIES} gfortran)