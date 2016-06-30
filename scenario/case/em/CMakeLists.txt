FILE(GLOB_RECURSE case_em_SRC ./*.cpp)
ADD_LIBRARY(case_em ${case_em_SRC})
TARGET_LINK_LIBRARIES(case_em simpla_lib)