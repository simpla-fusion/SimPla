
add_library(utilities  log.cpp properties.cpp ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(utilities ${Boost_LIBRARIES})
