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

#FILE(GLOB database_SRC  *.cpp *.h )

SET(database_SRC
        DataNodeExt.h
        DataNodeMemory.cpp
        DataNodeLua.cpp
        DataNodeHDF5.cpp
        DataNodeXDMF.cpp
        DataNodeIMAS.cpp
        LuaObject.cpp
        LuaObject.h
        LuaObjectExt.h
        )
ADD_LIBRARY(database ${database_SRC})
target_include_directories(database BEFORE PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${LUA_INCLUDE_DIR}
        ${MPI_C_INCLUDE_PATH}
        ${XDMF_INCLUDE_DIRS}
        #${VTK_INCLUDE_DIR}
        )
TARGET_LINK_LIBRARIES(database
        ${LUA_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${MPI_LIBRARIES}
        ${XDMF_LIBRARIES}
        #${VTK_LIBRARIES}
        )