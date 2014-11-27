ADD_EXECUTABLE(demo_kinetic demo_pic.cpp   )
TARGET_LINK_LIBRARIES(demo_kinetic use_case physics utilities io parallel)