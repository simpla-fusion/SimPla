ADD_SUBDIRECTORY( em  )
#ADD_SUBDIRECTORY( fluid  )
ADD_SUBDIRECTORY( field_fun  )
ADD_SUBDIRECTORY( io)
ADD_SUBDIRECTORY( preprocess)

add_library(modules  modules.cpp )
#TARGET_LINK_LIBRARIES(modules engine physics io  utilities  em)   