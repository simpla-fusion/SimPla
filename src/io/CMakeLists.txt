SET(IO_SRC IO.cpp
        HDF5Stream.cpp
#        XDMFStream.cpp
        IOStream.cpp)

link_directories(${HDF5_LIBRARY_DIRS})
add_library(io ${IO_SRC})
target_link_libraries(io ${HDF5_C_LIBRARIES} data_model)

