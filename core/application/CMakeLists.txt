add_library(application sp_object.cpp application.cpp application_main.cpp logo.cpp )
TARGET_LINK_LIBRARIES(application   physics  io    parallel utilities)