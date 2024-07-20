
FILE(GLOB gpu_maxwell_SRC ./*.cpp)

ADD_EXECUTABLE(gpu_maxwell ${gpu_maxwell_SRC})

find_package(Kokkos REQUIRED)

target_link_libraries(gpu_maxwell Kokkos::kokkos)
