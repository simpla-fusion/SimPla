
add_library(utilities   properties.cpp log.cpp parse_command_line.cpp config_parser.cpp  )
TARGET_LINK_LIBRARIES(utilities )

 
ADD_SUBDIRECTORY( test)
 