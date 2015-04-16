
ADD_SUBDIRECTORY( misc)

ADD_SUBDIRECTORY( general_field_op)
ADD_SUBDIRECTORY( mhd)
ADD_SUBDIRECTORY( em)
ADD_SUBDIRECTORY( probe_particle)
ADD_SUBDIRECTORY( pic)
ADD_SUBDIRECTORY( model)


ADD_EXECUTABLE(demo pic/demo_pic.cpp  em/demo_em.cpp )
TARGET_LINK_LIBRARIES(demo  simpla_aux)