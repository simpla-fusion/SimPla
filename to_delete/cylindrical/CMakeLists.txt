
FILE(GLOB_RECURSE demo_cylindrical_SRC ./*.cpp)

ADD_EXECUTABLE(demo_cylindrical ${demo_cylindrical_SRC})
target_compile_options(demo_cylindrical PUBLIC -DCARTESIAN_COORDINATE_SYTEM)
TARGET_LINK_LIBRARIES(demo_cylindrical simpla_lib simpla_main)

 

