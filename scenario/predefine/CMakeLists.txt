FILE(GLOB_RECURSE scenario_predefine_SRC ./*.cpp)
ADD_LIBRARY(scenario ${scenario_predefine_SRC})
TARGET_LINK_LIBRARIES(scenario simpla_lib)