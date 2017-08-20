#FIND_PACKAGE(VTK)
#MESSAGE("-- VTK version: ${VTK_VERSION}")
SET(XML2_DIR ${PROJECT_SOURCE_DIR}/external_project/xdmf)
SET(XDMF_DIR ${PROJECT_SOURCE_DIR}/external_project/xdmf)
FIND_PACKAGE(XDMF)


SET(HDF5_PREFER_PARALLEL OFF)
SET(HDF5_USE_STATIC_LIBRARIES OFF)
FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL REQUIRED)

FIND_PACKAGE(Lua 5.3 REQUIRED)
IF (LUA_FOUND)
    MESSAGE(STATUS "Lua version: ${LUA_INCLUDE_DIR} ")
ENDIF (LUA_FOUND)

FILE(GLOB data_backend_SRC  *.cpp *.h )

add_library(data_backend ${data_backend_SRC})
target_include_directories(data_backend BEFORE PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${LUA_INCLUDE_DIR}
        ${MPI_C_INCLUDE_PATH}
        ${XDMF_INCLUDE_DIRS}
        #${VTK_INCLUDE_DIR}
        )
target_link_libraries(data_backend
        parallel utilities

        ${LUA_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${MPI_LIBRARIES}
        ${XDMF_LIBRARIES}
        #${VTK_LIBRARIES}
        )