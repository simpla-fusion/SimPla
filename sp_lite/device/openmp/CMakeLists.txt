FILE(GLOB sp_lite_SRC_CPU ./*.c)

ADD_LIBRARY(sp_lite_openmp ${sp_lite_SRC_CPU})

target_compile_options(sp_lite_openmp  PUBLIC   -fopenmp  )
