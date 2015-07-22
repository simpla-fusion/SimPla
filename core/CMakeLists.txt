ADD_SUBDIRECTORY( gtl  )
ADD_SUBDIRECTORY( parallel )
ADD_SUBDIRECTORY( io)
ADD_SUBDIRECTORY( application )

ADD_SUBDIRECTORY( physics)

ADD_SUBDIRECTORY( mesh  )

ADD_SUBDIRECTORY( geometry )

ADD_SUBDIRECTORY( model )

ADD_SUBDIRECTORY( field )

ADD_SUBDIRECTORY( particle)

ADD_SUBDIRECTORY( solver  )

add_library(simpla_aux   )
TARGET_LINK_LIBRARIES(simpla_aux  application physics utilities io parallel gtl/dataset )