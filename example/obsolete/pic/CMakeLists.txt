ADD_EXECUTABLE(demo_pic demo_pic.cpp)
TARGET_LINK_LIBRARIES(demo_pic simpla_lib)

ADD_EXECUTABLE(collision_pic collision_pic.cpp)
TARGET_LINK_LIBRARIES(collision_pic simpla_lib)


ADD_EXECUTABLE(test_vi test_vi.cpp)
TARGET_LINK_LIBRARIES(test_vi simpla_lib)