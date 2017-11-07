FILE(GLOB modeling_SRC ./*.cpp)
ADD_EXECUTABLE(modeling ${modeling_SRC})
#SET_TARGET_PROPERTIES(tokamak PROPERTIES COMPILE_FLAGS " -xcuda ")
TARGET_LINK_LIBRARIES(modeling SimPlaMain
        -Wl,--whole-archive
        utilities algebra parallel geometry data data_backend predefine
        -Wl,--no-whole-archive
        ${OPENMP_LIBRARIES}        ${HDF5_LIBRARIES}

        )