#ADD_SUBDIRECTORY(case)
#ADD_SUBDIRECTORY(field_solver)
#ADD_SUBDIRECTORY(contexts)
#ADD_SUBDIRECTORY(fokker_planck)

FILE(GLOB_RECURSE predefine_SRC *.cpp )
add_library(predefine ${predefine_SRC} )
target_include_directories(predefine BEFORE PRIVATE ${OCE_INCLUDE_DIRS})
target_link_libraries(predefine  ${OCE_LIBRARIES}  )