

my_test(material_test    
        material_test.cpp 
     
      )
target_link_libraries(material_test physics mesh io)

ADD_EXECUTABLE(geqdsk_convert geqdsk.cpp geqdsk_xdmf.cpp geqdsk_convert.cpp) 
TARGET_LINK_LIBRARIES(geqdsk_convert   Xdmf io  ${MPI_LIBRARIES} )


add_library(utilities   geqdsk.cpp)
TARGET_LINK_LIBRARIES(utilities ${LUA_LIBRARIES}   io)
