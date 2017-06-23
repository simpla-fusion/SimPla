IF (NOT SAMRAI_DIR)
    IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        SET(SAMRAI_DIR /pkg/SAMRAI/3.11.2/pgi-17.4)
    ELSE ()
        SET(SAMRAI_DIR /pkg/SAMRAI/3.11.2/gcc-7.1.0)
    ENDIF ()

ENDIF (NOT SAMRAI_DIR)

FIND_PACKAGE(SAMRAI REQUIRED)

IF (SAMRAI_FOUND)
    MESSAGE("-- SAMRAI: " ${SAMRAI_INCLUDE_DIRS})
ENDIF (SAMRAI_FOUND)

INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})

#SET(OCCT_PREFER_COMMUNITY_EDITION ON)
#
#FIND_PACKAGE(OCE 0.17 REQUIRED)
#MESSAGE("-- OCE Found:${OCE_LIBRARIES} INCLUDE:${OCE_INCLUDE_DIRS}")
#
#IF (NOT MOAB_DIR)
#    SET(MOAB_DIR /pkg/sigma/1.2/lib)
#    SET(MeshKit_DIR /pkg/sigma/1.2/lib)
#ENDIF (NOT MOAB_DIR)
#
#
#FIND_PACKAGE(MOAB REQUIRED)
#FIND_PACKAGE(MeshKit REQUIRED)
#IF (MOAB_FOUND)
#    MESSAGE(" -- MOAB Found: ${MOAB_LIBRARIES};  INCLUDE: ${MOAB_INCLUDE_DIRS}")
#ENDIF (MOAB_FOUND)
#
#IF (MeshKit_FOUND)
#    MESSAGE(" -- MeshKit Found: ${MeshKit_LIBRARIES};  INCLUDE: ${MESHKIT_INCLUDE_DIRS}")
#ENDIF (MeshKit_FOUND)
#
# ${MOAB_INCLUDE_DIRS} ${MESHKIT_INCLUDE_DIRS} ${OCE_INCLUDE_DIRS}


FILE(GLOB third_part_SRC *.cpp)
#FILE(GLOB third_part_SRC DataTable.cpp )
add_library(third_part ${third_part_SRC})

TARGET_LINK_LIBRARIES(third_part
        #     ${OCE_LIBRARIES}   ${MeshKit_LIBRARIES} ${MOAB_LIBRARIES}
        ${SAMRAI_LIBRARIES} netcdf ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}

        #        toolbox geometry parallel data mesh engine model physics
        )
target_include_directories(third_part BEFORE PRIVATE  ${MPI_C_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})

#FILE(GLOB amr_test_SRC ./*.cpp)
#
#add_executable(amr_test ${amr_test_SRC})
#
#TARGET_LINK_LIBRARIES(amr_test
#        #     ${OCE_LIBRARIES}   ${MeshKit_LIBRARIES} ${MOAB_LIBRARIES}
#        ${SAMRAI_LIBRARIES} netcdf
#        ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}
#        toolbox geometry parallel data mesh engine model physics SimPlaMAIN
#        )