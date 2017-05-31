FILE(GLOB main_SRC ./*.cpp)
add_library(SimPlaMain ${main_SRC})
TARGET_LINK_LIBRARIES(SimPlaMain ${SIMPLA_LIBRARIES})
SET(SIMPLA_MAIN SimPlaMain)