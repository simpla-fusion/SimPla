add_library(use_case  use_case.cpp use_case_main.cpp logo.cpp )
 
TARGET_LINK_LIBRARIES(use_case    physics  io     parallel utilities)