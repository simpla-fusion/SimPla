INCLUDE_DIRECTORIES(/pkg/google_benchmark/current/include)
LINK_DIRECTORIES(/pkg/google_benchmark/current/lib)


INCLUDE(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_LIST_DIR})
FOREACH (subdir ${SUBDIRS})
    MESSAGE("Add sub directory: ${subdir}")
    ADD_SUBDIRECTORY(${subdir})
ENDFOREACH ()


#add_subdirectory(engine)
#add_subdirectory(algebra)
#add_subdirectory(model)
#add_subdirectory(concept)
#add_subdirectory(data)