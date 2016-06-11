ADD_EXECUTABLE(demo_em demo_em.cpp Boris.c)
target_compile_options(demo_em PUBLIC -DCARTESIAN_COORDINATE_SYTEM)
TARGET_LINK_LIBRARIES(demo_em simpla_lib)



