FILE(GLOB io_SRC ./*.cpp)
add_library(io ${io_SRC})
target_link_libraries(io ${HDF5_LIBRARIES})