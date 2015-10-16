ADD_EXECUTABLE(demo_em  demo_em.cpp   )
target_compile_options(demo_em PUBLIC   -DCARTESIAN_COORDINATE_SYTEM )
TARGET_LINK_LIBRARIES(demo_em  simpla_aux)

ADD_EXECUTABLE(demo_em_cylindrical  demo_em.cpp   )
target_compile_options(demo_em_cylindrical PUBLIC   -DCYLINDRICAL_COORDINATE_SYTEM  )
TARGET_LINK_LIBRARIES(demo_em_cylindrical  simpla_aux)
