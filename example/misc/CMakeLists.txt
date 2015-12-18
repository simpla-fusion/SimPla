ADD_EXECUTABLE(demo_misc  demo_misc.cpp   )
#TARGET_LINK_LIBRARIES(demo_misc  task_flow physics utilities io parallel DataModel)

ADD_EXECUTABLE(demo_misc2  demo_misc2.cpp   )
TARGET_LINK_LIBRARIES(demo_misc2  simpla_aux)

ADD_EXECUTABLE(demo_misc3  demo_misc3.cpp   )
TARGET_LINK_LIBRARIES(demo_misc3  simpla_aux)