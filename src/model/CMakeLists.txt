

my_test(material_test    
        material_test.cpp 
     
      )
target_link_libraries(material_test physics mesh io)

ADD_EXECUTABLE(geqdsk_test geqdsk.cpp geqdsk_test.cpp) 
TARGET_LINK_LIBRARIES(geqdsk_test   Xdmf io)


add_library(utilities   geqdsk.cpp)
TARGET_LINK_LIBRARIES(utilities ${LUA_LIBRARIES}   io)
