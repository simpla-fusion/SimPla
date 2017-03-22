
FILE(GLOB data_backend_SRC ./*.cpp)

add_library(data_backend ${data_backend_SRC})
target_link_libraries(data_backend ${LUA_LIBRARIES} parallel toolbox  ${HDF5_LIBRARIES} ${MPI_LIBRARIES}  )