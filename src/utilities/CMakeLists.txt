ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test   ${LUA_LIBRARIES}  )

ADD_EXECUTABLE(geqdsk_test geqdsk.cpp geqdsk_test.cpp ../io/write_hdf5.cpp)
TARGET_LINK_LIBRARIES(geqdsk_test   ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES}  ${HDF5_HL_LIBRARIES}  hdf5_cpp hdf5_hl_cpp hdf5_hl)