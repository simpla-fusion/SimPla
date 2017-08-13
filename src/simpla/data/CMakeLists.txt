


add_subdirectory(backend)

FILE(GLOB data_SRC  *.cpp  )
add_library(data ${data_SRC})
target_link_libraries(data   parallel utilities data_backend )