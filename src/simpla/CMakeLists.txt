INCLUDE(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
FOREACH (subdir ${SUBDIRS})
    MESSAGE("Add sub directory: ${subdir}")
    ADD_SUBDIRECTORY(${subdir})
ENDFOREACH ()



#TARGET_LINK_LIBRARIES(SimPlaLib
#        netcdf ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${TBB_LIBRARIES}
#          utilities geometry parallel data mesh engine geometry physics third_part
#        )

