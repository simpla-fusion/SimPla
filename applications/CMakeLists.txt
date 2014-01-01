
ADD_SUBDIRECTORY( solver )
ADD_SUBDIRECTORY( contexts )
              
ADD_EXECUTABLE(simpla  simpla.cpp   )

TARGET_LINK_LIBRARIES(simpla physics explicit_em field_solver ${LUA_LIBRARIES} io)

