ADD_SUBDIRECTORY(gtl)
ADD_SUBDIRECTORY(parallel)
ADD_SUBDIRECTORY(io)
ADD_SUBDIRECTORY(application)
ADD_SUBDIRECTORY(dataset)

ADD_SUBDIRECTORY(physics)

ADD_SUBDIRECTORY(manifold)

ADD_SUBDIRECTORY(geometry)

ADD_SUBDIRECTORY(model)

ADD_SUBDIRECTORY(field)

ADD_SUBDIRECTORY(particle)


add_library(simpla_aux)
TARGET_LINK_LIBRARIES(simpla_aux application physics utilities io parallel dataset)