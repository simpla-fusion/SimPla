#FIND_PACKAGE(SAMRAI)

INCLUDE(ExternalProject)

#IF (NOT SAMRAI_FOUND)
ExternalProject_Add(SAMRAI
        PREFIX ${PROJECT_BINARY_DIR}/external_project/SAMRAI
        #        URL https://github.com/LLNL/SAMRAI/archive/master.zip
        URL file:///home/salmon/workspace/archive/samrai_master.zip
        CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        BUILD_COMMAND ${MAKE}
        LOG_DOWNLOAD 1
        LOG_UPDATE 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_TEST 1
        LOG_INSTALL 1
        )
ExternalProject_Get_Property(SAMRAI SOURCE_DIR)
ExternalProject_Get_Property(SAMRAI BINARY_DIR)
SET(SAMRAI_SOURCE_DIR ${SOURCE_DIR})
SET(SAMRAI_BINARY_DIR ${BINARY_DIR})

MESSAGE(SAMRAI_SOURCE_DIR = ${SAMRAI_SOURCE_DIR})
MESSAGE(SAMRAI_BINARY_DIR = ${SAMRAI_BINARY_DIR})

#FIND_PACKAGE(SAMRAI)
#ENDIF (NOT SAMRAI_FOUND)
INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})
MESSAGE(STATUS "SAMRAI is included" ${SAMRAI_BINARY_DIR})
FILE(GLOB SAMRAIWrapper_SRC ./*.cpp)
add_library(samrai_wrapper ${SAMRAIWrapper_SRC})
TARGET_LINK_LIBRARIES(samrai_wrapper ${SAMRAI_LIBRARIES})
SET(MESH_WEAPPER samrai_wrapper)
add_dependencies(samrai_wrapper SAMRAI)
