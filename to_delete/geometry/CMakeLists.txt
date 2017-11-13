FILE(GLOB GeometryModel_src ./*.cpp)
ADD_EXECUTABLE(GeometryModel ${GeometryModel_src})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(GeometryModel
        -Wl,--whole-archive
        utilities    geometry  geometry_backend data data_backend
        -Wl,--no-whole-archive
        )