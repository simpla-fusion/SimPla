
ADD_SUBDIRECTORY( field_solver ) 
ADD_SUBDIRECTORY( contexts )
              
ADD_EXECUTABLE(simpla  simpla.cpp   )

TARGET_LINK_LIBRARIES(simpla  physics context  utilities io)

