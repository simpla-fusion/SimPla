ADD_EXECUTABLE(demo_model demo_model.cpp)
TARGET_LINK_LIBRARIES(demo_model    application parallel utilities io   dataset)