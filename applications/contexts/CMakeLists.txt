add_library(explicit_em explicit_em.cpp)
TARGET_LINK_LIBRARIES(explicit_em physics  ${LUA_LIBRARIES} io)

add_library(trajectory trajectory.cpp)
TARGET_LINK_LIBRARIES(trajectory physics  ${LUA_LIBRARIES} io)