
FILE(GLOB_RECURSE demo_mb_SRC ./*.cpp)

ADD_EXECUTABLE(demo_mb ${demo_mb_SRC})
target_compile_options(demo_mb PUBLIC -DCARTESIAN_COORDINATE_SYTEM)
TARGET_LINK_LIBRARIES(demo_mb simpla_lib simpla_main)

 

