add_subdirectory(db)

FILE(GLOB   data_SRC  *.cpp *.h  )
add_library(data ${data_SRC})
target_link_libraries(data
        data_backend
        parallel utilities    )
