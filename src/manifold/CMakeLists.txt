#add_subdirectory(structured )
#add_subdirectory(unstructured )
#add_subdirectory(meshfree)
#add_subdirectory(policy)
add_subdirectory(test)


FILE(GLOB_RECURSE manifold_SRC ./*.cpp)
add_library(manifold ${manifold_SRC})
