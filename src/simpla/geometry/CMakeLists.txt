

FILE(GLOB geometry_SRC  *.cpp *.h  )
add_library(geometry ${geometry_SRC})

#add_subdirectory(occ)

TARGET_LINK_LIBRARIES(geometry    utilities data
#        geometry_occ
        )

#FILE(GLOB geometry_SRC ./*.cpp)
#add_library(geometry ${geometry_SRC} Cube.cpp CoordinateSystem.cpp)
