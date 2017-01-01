INCLUDE_DIRECTORIES(/pkg/google_benchmark/current/include)
LINK_DIRECTORIES(/pkg/google_benchmark/current/lib)


add_executable(nTuplePerf nTuplePerf.cpp)
target_link_libraries(nTuplePerf benchmark)