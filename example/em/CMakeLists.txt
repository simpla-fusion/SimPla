ADD_EXECUTABLE(demo_em demo_em.cpp)
target_compile_options(demo_em PUBLIC -DCARTESIAN_COORDINATE_SYTEM)
TARGET_LINK_LIBRARIES(demo_em simpla_lib)

ADD_EXECUTABLE(em em.cpp)
TARGET_LINK_LIBRARIES(em simpla_lib)


ADD_EXECUTABLE(em_plasma em_plasma.cpp)

TARGET_LINK_LIBRARIES(em_plasma simpla_lib)


ADD_EXECUTABLE(em_task em_task.cpp)
TARGET_LINK_LIBRARIES(em_task  simpla_lib)
