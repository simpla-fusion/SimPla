add_library(explicit_em explicit_em.cpp)
TARGET_LINK_LIBRARIES(explicit_em physics  ${LUA_LIBRARIES} io)

