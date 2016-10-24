
SET(SAMRAI_BUILD_DIR /home/salmon/workspace/build/SAMRAI)
SET(SAMRAI_SOURCE_DIR /home/salmon/workspace/SAMRAI-master)
FIND_PACKAGE(SAMRAI)

IF (SAMRAI_FOUND)
    INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})
    MESSAGE(STATUS "SAMRAI is included")
    FILE(GLOB SAMRAIWrapper_SRC ./*.cpp)
    add_library(samrai_wrapper ${SAMRAIWrapper_SRC})
    TARGET_LINK_LIBRARIES(samrai_wrapper ${SAMRAI_LIBRARIES})
    SET(MESH_WEAPPER samrai_wrapper)
ENDIF (SAMRAI_FOUND)