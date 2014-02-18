add_library(context context.cpp)
TARGET_LINK_LIBRARIES(context physics  ${LUA_LIBRARIES} io)

