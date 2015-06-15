ADD_EXECUTABLE(demo_em  demo_em.cpp   )
TARGET_LINK_LIBRARIES(demo_em  simpla_aux)

ADD_EXECUTABLE(demo_em_cylindrical  demo_em.cpp   )
target_compile_options(demo_em_cylindrical PUBLIC   -DCOORDINATE_SYSTEM=cylindrical)
TARGET_LINK_LIBRARIES(demo_em_cylindrical  simpla_aux)
