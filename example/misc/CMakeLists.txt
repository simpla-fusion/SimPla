ADD_EXECUTABLE(demo_misc  demo_misc.cpp   )
#TARGET_LINK_LIBRARIES(demo_misc  application physics utilities io parallel dataset)

ADD_EXECUTABLE(demo_misc2  demo_misc2.cpp   )
TARGET_LINK_LIBRARIES(demo_misc2  simpla_aux)

ADD_EXECUTABLE(demo_misc3  demo_misc3.cpp   )
TARGET_LINK_LIBRARIES(demo_misc3  simpla_aux)