ADD_EXECUTABLE(MPIUpdater_test MPIUpdater_test.cpp)
target_link_libraries(MPIUpdater_test
        -Wl,--whole-archive
        data utilities data_backend parallel
        -Wl,--no-whole-archive
        )
ADD_EXECUTABLE(mpi_string_dummy mpi_string_dummy.cpp)
target_link_libraries(mpi_string_dummy parallel utilities)
