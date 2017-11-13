FILE(GLOB modeling_SRC ./*.cpp
        ../../src/simpla/predefine/device/Tokamak.h
        ../../src/simpla/predefine/device/Tokamak.cpp
        )
ADD_EXECUTABLE(modeling ${modeling_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")

TARGET_LINK_LIBRARIES(modeling
        -Wl,--whole-archive
        utilities geometry geometry_backend data data_backend
        -Wl,--no-whole-archive
        )