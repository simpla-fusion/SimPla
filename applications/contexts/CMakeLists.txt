add_library(context context.cpp)
TARGET_LINK_LIBRARIES(context physics  mesh ${LUA_LIBRARIES} io)

