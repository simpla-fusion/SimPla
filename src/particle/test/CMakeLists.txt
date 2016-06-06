#simpla_test(kinetic_particle_test kinetic_particle_test.cpp)
#target_link_libraries(kinetic_particle_test physics io parallel utilities data_model)
#
#simpla_test(probe_particle_test probe_particle_test.cpp)
#target_link_libraries(probe_particle_test physics io parallel utilities)
#
ADD_EXECUTABLE(particle_dummy particle_dummy.c)
target_link_libraries(particle_dummy particle)
