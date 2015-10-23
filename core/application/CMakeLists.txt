SET(APPLICATION_SRC sp_object.cpp application.cpp application_main.cpp)

SET(USE_CASE_SRC use_case.cpp use_case_main.cpp)

add_library(application ${APPLICATION_SRC})
